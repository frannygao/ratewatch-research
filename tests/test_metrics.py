"""Checks on the measurement layer."""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from ratewatch import record
from ratewatch.metrics import (binomial_p, bootstrap_ci, brier_skill, by_meeting_type,
                               evaluate, mcnemar, meeting_type, scores)
from ratewatch.model import Forecast

CUT, HOLD, HIKE = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def prediction(date, actual, previous, model, persistence=None):
    return {'meeting_date': date, 'actual': actual, 'previous_label': previous,
            'model': model, 'persistence': persistence or HOLD}


class ScoreTests(unittest.TestCase):
    def test_a_perfect_confident_forecast(self):
        result = scores(['cut', 'hold'], [CUT, HOLD])
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['brier'], 0.0)

    def test_a_confidently_wrong_forecast_scores_two(self):
        self.assertEqual(scores(['cut'], [HIKE])['brier'], 2.0)

    def test_log_loss_is_withheld_for_zero_one_forecasts(self):
        # a 0/100 forecast's penalty depends on the log floor, not the forecast
        self.assertIsNone(scores(['cut', 'hold'], [CUT, HIKE])['log_loss'])
        self.assertIsNotNone(scores(['cut'], [[0.5, 0.3, 0.2]])['log_loss'])

    def test_balanced_accuracy_ignores_classes_that_never_occurred(self):
        # only holds happened, so the mean recall is over that one class
        result = scores(['hold', 'hold'], [HOLD, HOLD])
        self.assertEqual(result['balanced_accuracy'], 1.0)
        self.assertIsNone(result['recall']['cut'])

    def test_balanced_accuracy_punishes_ignoring_a_rare_class(self):
        # three holds and one cut, all predicted hold: 100% on hold, 0% on cut
        result = scores(['hold'] * 3 + ['cut'], [HOLD] * 4)
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['balanced_accuracy'], 0.5)

    def test_an_even_forecast_scores_the_log_of_three(self):
        even = [1 / 3, 1 / 3, 1 / 3]
        self.assertAlmostEqual(scores(['cut'], [even])['log_loss'], 1.0986, places=3)


class MeetingTypeTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(meeting_type('hold', 'hold'), 'unchanged')
        self.assertEqual(meeting_type('hold', 'cut'), 'new move')
        self.assertEqual(meeting_type('cut', 'hike'), 'new move')
        self.assertEqual(meeting_type('cut', 'hold'), 'returned to hold')

    def test_a_missing_previous_decision_counts_as_unchanged(self):
        # nothing to have changed from, and pandas may hand this over as NaN
        self.assertEqual(meeting_type(None, 'cut'), 'unchanged')
        self.assertEqual(meeting_type(float('nan'), 'cut'), 'unchanged')

    def test_split_separates_the_meetings_that_mattered(self):
        predictions = [prediction('2020-01-01', 'hold', 'hold', HOLD),
                       prediction('2020-03-01', 'cut', 'hold', HOLD),     # missed a new move
                       prediction('2020-05-01', 'cut', 'cut', CUT),
                       prediction('2020-07-01', 'hold', 'cut', HOLD)]
        split = by_meeting_type(predictions, 'model')
        self.assertEqual(split['new move'], {'n': 1, 'correct': 0, 'accuracy': 0.0,
                                             'meetings': ['2020-03-01']})
        self.assertEqual(split['unchanged']['accuracy'], 1.0)
        self.assertEqual(split['returned to hold']['accuracy'], 1.0)


class SignificanceTests(unittest.TestCase):
    def test_exact_binomial_matches_a_hand_calculation(self):
        # 6 of 14, two-sided: 2 * sum(C(14,0..6)) / 2**14 = 2 * 6476 / 16384
        self.assertAlmostEqual(binomial_p(6, 14), 2 * 6476 / 16384, places=10)
        self.assertEqual(binomial_p(0, 0), 1.0)
        self.assertAlmostEqual(binomial_p(0, 10), 2 / 1024, places=10)

    def test_mcnemar_only_counts_disagreements(self):
        predictions = [prediction('a', 'cut', 'hold', CUT, persistence=CUT),   # both right
                       prediction('b', 'cut', 'hold', CUT, persistence=HOLD),  # model wins
                       prediction('c', 'hold', 'hold', CUT, persistence=HOLD)] # model loses
        test = mcnemar(predictions, 'model', 'persistence')
        self.assertEqual((test['wins'], test['losses']), (1, 1))
        self.assertEqual(test['p_value'], 1.0)


class SkillTests(unittest.TestCase):
    def test_skill_is_zero_against_itself_and_one_when_perfect(self):
        predictions = [prediction('a', 'cut', 'hold', CUT, persistence=HOLD),
                       prediction('b', 'hold', 'hold', HOLD, persistence=HOLD)]
        self.assertAlmostEqual(brier_skill(predictions, 'persistence', 'persistence'), 0.0)
        self.assertAlmostEqual(brier_skill(predictions, 'model', 'persistence'), 1.0)

    def test_skill_goes_negative_when_worse(self):
        predictions = [prediction('a', 'hold', 'hold', CUT, persistence=HOLD),   # model wrong
                       prediction('b', 'cut', 'hold', HOLD, persistence=HOLD)]   # both wrong
        self.assertAlmostEqual(brier_skill(predictions, 'model', 'persistence'), -1.0)

    def test_skill_is_undefined_against_a_flawless_reference(self):
        # nothing to improve on, so there is no share of error removed to report
        predictions = [prediction('a', 'hold', 'hold', CUT, persistence=HOLD)]
        self.assertIsNone(brier_skill(predictions, 'model', 'persistence'))


class IntervalTests(unittest.TestCase):
    def test_the_interval_brackets_the_point_estimate(self):
        predictions = [prediction(f'{2000 + i}-01-01', 'hold' if i % 4 else 'cut', 'hold',
                                  HOLD if i % 4 else CUT) for i in range(40)]
        interval = bootstrap_ci(predictions, 'model', draws=500)
        self.assertLessEqual(interval['low'], interval['point'])
        self.assertGreaterEqual(interval['high'], interval['point'])

    def test_evaluate_returns_every_block(self):
        predictions = [prediction(f'{2000 + i}-01-01', 'hold' if i % 4 else 'cut', 'hold',
                                  HOLD if i % 4 else CUT) for i in range(40)]
        result = evaluate(predictions, ['model', 'persistence'])
        self.assertEqual(result.n, 40)
        for method in ['model', 'persistence']:
            self.assertIn(method, result.scores)
            self.assertIn(method, result.by_meeting_type)
            self.assertIn(method, result.skill)
            self.assertIn(method, result.accuracy_ci)
        # the reference is not tested against itself
        self.assertNotIn('persistence', result.significance)
        self.assertEqual(result.accuracy(), result.scores['model']['accuracy'])
        self.assertIn('predictions', result.as_dict())

    def test_moves_called_reads_the_new_move_group(self):
        predictions = [prediction('2020-01-01', 'hold', 'hold', HOLD),
                       prediction('2020-03-01', 'cut', 'hold', HOLD),
                       prediction('2020-05-01', 'hike', 'hold', HIKE)]
        result = evaluate(predictions, ['model', 'persistence'])
        self.assertEqual(result.moves_called(), (1, 2))


class RecordTests(unittest.TestCase):
    def forecast(self, meeting_date):
        return Forecast(
            meeting_date=meeting_date,
            meeting_cutoff=f'{meeting_date}T16:00:00-04:00',
            information_cutoff='2026-09-01T00:00:00+00:00',
            probabilities={'cut': 0.2, 'hold': 0.7, 'hike': 0.1},
            features={}, used={},
            baselines={'persistence': {'cut': 0.0, 'hold': 1.0, 'hike': 0.0}},
            trained_on=100)

    def test_forecast_reports_its_leading_decision(self):
        self.assertEqual(self.forecast('2026-10-28').leading, 'hold')

    def future_meeting(self):
        return (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat()

    def test_adds_then_refuses_a_second_forecast_for_the_same_meeting(self):
        with TemporaryDirectory() as folder:
            date = self.future_meeting()
            record.add(folder, self.forecast(date), 'meeting-v1')
            self.assertEqual(len(record.read(folder)), 1)
            with self.assertRaises(ValueError):
                record.add(folder, self.forecast(date), 'meeting-v1')
            # a different protocol is a different track record, so it is allowed
            record.add(folder, self.forecast(date), 'meeting-v2')
            self.assertEqual(len(record.read(folder)), 2)

    def test_refuses_a_forecast_for_a_meeting_already_decided(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                record.add(folder, self.forecast('2020-01-22'), 'meeting-v1')

    def test_scores_only_meetings_that_have_happened(self):
        with TemporaryDirectory() as folder:
            date = self.future_meeting()
            record.add(folder, self.forecast(date), 'meeting-v1')
            pending = record.score(folder, [{'date': date, 'label': None}])
            self.assertEqual(pending['meeting-v1'], {'issued': 1, 'graded': 0, 'scores': None})
            graded = record.score(folder, [{'date': date, 'label': 'hold'}])
            self.assertEqual(graded['meeting-v1']['graded'], 1)
            self.assertEqual(graded['meeting-v1']['scores']['model']['accuracy'], 1.0)
            self.assertAlmostEqual(graded['meeting-v1']['scores']['model']['log_loss'], 0.3567, places=3)

    def test_an_empty_log_reads_as_empty(self):
        with TemporaryDirectory() as folder:
            self.assertEqual(record.read(Path(folder) / 'missing'), [])
            self.assertEqual(record.score(folder, []), {})


if __name__ == '__main__':
    unittest.main()
