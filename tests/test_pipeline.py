"""Timing, feature and leakage checks on the data-to-prediction path."""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ratewatch.features import (TORONTO, build_panel, cutoff_for, feature_row,
                                latest_release, next_meeting, previous_label)
from ratewatch.model import backtest, baselines


def monthly(start, values):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq='MS'))


def sample_series():
    """Seven years of monthly figures, plus a daily policy rate that steps up once."""
    months = 84
    cpi = monthly('2020-01-01', [100 + i for i in range(months)])
    # 2026-06 is month 77; step unemployment over the three months ending there
    unemployment = monthly('2020-01-01', [6.0] * 75 + [6.1, 6.2, 6.3] + [6.0] * (months - 78))
    days = pd.date_range('2020-01-01', '2026-12-31', freq='D')
    rate = pd.Series([2.0 if day < pd.Timestamp('2026-06-01') else 3.0 for day in days], index=days)
    return {'cpi': cpi, 'unemployment': unemployment, 'policy': rate}


def sample_releases(reference='2026-06-01', published='2026-07-15'):
    day = datetime.fromisoformat(published).date()
    return [{'series': series, 'reference': reference, 'published': published,
             'available_after': (day + timedelta(days=1)).isoformat(), 'url': 'x'}
            for series in ('cpi', 'unemployment')]


class CutoffTests(unittest.TestCase):
    def test_steps_back_over_a_weekend(self):
        # monday 2026-01-05: the day before is a sunday, so the cutoff is friday
        self.assertEqual(cutoff_for('2026-01-05').date().isoformat(), '2026-01-02')

    def test_plain_weekday(self):
        self.assertEqual(cutoff_for('2026-01-08').date().isoformat(), '2026-01-07')

    def test_is_four_pm_toronto(self):
        stamp = cutoff_for('2026-06-10')
        self.assertEqual((stamp.hour, stamp.minute), (16, 0))
        self.assertEqual(stamp.tzinfo, TORONTO)


class ReleaseTests(unittest.TestCase):
    def test_a_figure_published_on_the_cutoff_day_is_not_used(self):
        releases = sample_releases(published='2026-07-15')
        cutoff = cutoff_for('2026-07-16')          # cutoff falls on 2026-07-15
        with self.assertRaises(ValueError):
            latest_release(releases, 'cpi', cutoff)

    def test_a_figure_published_the_day_before_is_used(self):
        releases = sample_releases(published='2026-07-14')
        found = latest_release(releases, 'cpi', cutoff_for('2026-07-16'))
        self.assertEqual(found['reference'], '2026-06-01')

    def test_the_most_recent_reference_month_wins(self):
        releases = sample_releases('2026-05-01', '2026-06-15') + sample_releases('2026-06-01', '2026-07-14')
        found = latest_release(releases, 'cpi', cutoff_for('2026-07-16'))
        self.assertEqual(found['reference'], '2026-06-01')


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.series = sample_series()
        self.releases = sample_releases()
        self.values, self.used = feature_row(self.series, self.releases, cutoff_for('2026-07-20'))

    def test_real_rate_and_inflation_gap_agree_with_the_policy_rate(self):
        # real_rate = rate - inflation and inflation_gap = inflation - 2,
        # so the two must sum to rate - 2 whatever the inflation rate is.
        self.assertAlmostEqual(self.values['real_rate'] + self.values['inflation_gap'],
                               self.used['policy_rate'] - 2)

    def test_policy_momentum_sees_the_step(self):
        # the rate went 2.0 -> 3.0 on 2026-04-01, within three months of the cutoff
        self.assertAlmostEqual(self.values['policy_momentum'], 1.0)

    def test_unemployment_change_is_the_three_month_difference(self):
        self.assertAlmostEqual(self.values['unemployment_change'], 0.3)

    def test_records_what_it_used(self):
        self.assertEqual(self.used['cpi']['reference'], '2026-06-01')
        self.assertEqual(self.used['policy_rate'], 3.0)

    def test_rejects_a_naive_cutoff(self):
        with self.assertRaises(ValueError):
            feature_row(self.series, self.releases, datetime(2026, 7, 20, 16))

    def test_rejects_a_stale_release(self):
        with self.assertRaises(ValueError):
            feature_row(self.series, sample_releases('2025-01-01', '2025-02-15'),
                        cutoff_for('2026-07-20'))


class PanelTests(unittest.TestCase):
    def test_records_the_previous_decision_and_drops_what_it_cannot_build(self):
        meetings = [{'date': '2019-01-09', 'label': 'hold'},      # before any release
                    {'date': '2026-07-20', 'label': 'cut'},
                    {'date': '2026-07-22', 'label': 'hold'},
                    {'date': '2027-01-20', 'label': None}]
        panel, dropped = build_panel(sample_series(), sample_releases(), meetings)
        self.assertEqual(list(panel['meeting_date']), ['2026-07-20', '2026-07-22'])
        self.assertIsNone(previous_label(panel.iloc[0]))
        self.assertEqual(previous_label(panel.iloc[1]), 'cut')
        self.assertEqual([d['meeting'] for d in dropped], ['2019-01-09'])

    def test_next_meeting_is_the_soonest_undecided_one(self):
        meetings = [{'date': '2026-07-20', 'label': 'cut'},
                    {'date': '2026-10-28', 'label': None},
                    {'date': '2026-12-09', 'label': None}]
        now = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.assertEqual(next_meeting(meetings, now)['date'], '2026-10-28')
        self.assertIsNone(next_meeting(meetings[:1], now))


class BacktestTests(unittest.TestCase):
    def build(self, labels):
        rows = []
        previous = None
        for i, label in enumerate(labels):
            rows.append({'meeting_date': f'{2000 + i // 8}-{i % 8 + 1:02d}-10',
                         'cutoff': f'{2000 + i // 8}-{i % 8 + 1:02d}-09T16:00:00-05:00',
                         'label': label, 'previous_label': previous,
                         'inflation_gap': i % 5 - 2, 'inflation_trend': i % 3 - 1,
                         'unemployment_change': i % 4 - 2, 'unemployment_gap': i % 7 - 3,
                         'real_rate': i % 6 - 3, 'policy_momentum': i % 3 - 1})
            previous = label
        return pd.DataFrame(rows)

    def test_never_trains_on_the_meeting_it_is_predicting_or_later(self):
        labels = (['cut', 'hold', 'hike'] * 16)[:45]
        panel = self.build(labels)
        predictions = backtest(panel)
        self.assertTrue(predictions)
        for prediction in predictions:
            day = prediction['cutoff'][:10]
            trained = panel[panel['meeting_date'] < day]
            self.assertEqual(len(trained), prediction['trained_on'])
            self.assertTrue((trained['meeting_date'] < day).all())
            self.assertAlmostEqual(sum(prediction['model']), 1.0)

    def test_changing_the_last_meeting_leaves_earlier_predictions_alone(self):
        labels = (['cut', 'hold', 'hike'] * 16)[:45]
        panel = self.build(labels)
        original = backtest(panel)
        altered = panel.copy()
        altered.loc[altered.index[-1], 'real_rate'] = 999
        self.assertEqual(original[:-1], backtest(altered)[:-1])

    def test_baselines_only_look_backwards(self):
        result = baselines(['cut', 'cut', 'hold', 'cut'])
        self.assertEqual(result['persistence'], [1.0, 0.0, 0.0])
        self.assertEqual(result['always_hold'], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(result['frequency']), 1.0)
        self.assertAlmostEqual(sum(result['transition']), 1.0)


if __name__ == '__main__':
    unittest.main()


class FigureTests(unittest.TestCase):
    """The figures are part of the deliverable, so check they actually render."""

    def evaluation(self):
        from ratewatch.config import METHODS
        from ratewatch.metrics import evaluate
        labels = (['hold'] * 3 + ['cut', 'cut', 'hold', 'hike']) * 6
        predictions, previous = [], None
        for i, label in enumerate(labels):
            spread = {'cut': [.6, .3, .1], 'hold': [.2, .6, .2], 'hike': [.1, .3, .6]}[label]
            predictions.append({'meeting_date': f'{2010 + i // 8}-{i % 8 + 1:02d}-10',
                                'actual': label, 'previous_label': previous,
                                'model': spread, 'frequency': [.2, .6, .2],
                                'transition': [.3, .4, .3], 'always_hold': [0., 1., 0.],
                                'persistence': [float(x == previous) for x
                                                in ['cut', 'hold', 'hike']]
                                if previous else [0., 1., 0.]})
            previous = label
        return evaluate(predictions, METHODS)

    def test_every_figure_renders(self):
        from ratewatch.viz import draw_all
        panel, _ = build_panel(sample_series(), sample_releases(),
                               [{'date': '2026-07-20', 'label': 'cut'},
                                {'date': '2026-07-22', 'label': 'hold'}])
        with TemporaryDirectory() as folder:
            paths = draw_all(self.evaluation(), panel, sample_series(), Path(folder))
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 5000, f'{path.name} looks empty')
                self.assertEqual(path.suffix, '.png')
