import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ratewatch.audit import issue_forecast, read_log, score_log
from ratewatch.forecast import (
    evaluate_meetings, feature_row, features, fit_model, forecast_next,
    predict_probabilities, training,
)
from ratewatch.sources import Download, refresh_sources, release_calendar_url


class ForecastTests(unittest.TestCase):
    def test_feature_values(self):
        months = pd.date_range('2020-01-01', '2026-12-01', freq='MS')
        days = pd.date_range('2020-01-01', '2026-12-31')
        raw = {'cpi': pd.Series(100 * 1.002 ** np.arange(len(months)), index=months),
               'unemployment': pd.Series(5 + 0.1 * np.arange(len(months)), index=months),
               'policy': pd.Series(3.0, index=days)}
        releases = [{'series': series, 'reference': '2026-07-01',
                     'published_date': '2026-08-17', 'available_after': '2026-08-18'}
                    for series in ['cpi', 'unemployment']]
        values, _ = feature_row(raw, releases, datetime.fromisoformat('2026-09-01T16:00:00-04:00'))
        inflation = (1.002 ** 12 - 1) * 100
        expected = {'inflation_gap': inflation - 2, 'inflation_trend': 0,
                    'unemployment_change': 0.3, 'unemployment_gap': 1.75,
                    'real_rate': 3 - inflation, 'policy_momentum': 0}
        for name, value in expected.items():
            self.assertAlmostEqual(values[name], value)
        with self.assertRaises(ValueError):
            feature_row(raw, releases, datetime(2026, 9, 1))
        with self.assertRaises(ValueError):
            feature_row(raw, releases, datetime.fromisoformat('2026-12-01T16:00:00-05:00'))

    def test_probabilities_follow_named_order(self):
        model = Mock()
        model.classes_ = np.array(['hike', 'cut', 'hold'])
        model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        result = predict_probabilities(model, dict.fromkeys(features, 0.0))
        self.assertEqual(list(result), ['cut', 'hold', 'hike'])
        self.assertEqual(result, {'cut': 0.2, 'hold': 0.7, 'hike': 0.1})

    def test_training_uses_toronto_day(self):
        panel = pd.DataFrame({'meeting_date': ['2026-09-01', '2026-09-02']})
        self.assertTrue(training(panel, '2026-09-02T01:00:00+00:00').empty)

    def test_walk_forward_fits_only_prior_meetings(self):
        dates = pd.date_range('2020-01-01', periods=44, freq='MS')
        panel = pd.DataFrame({feature: np.arange(44) / 10 for feature in features})
        panel['meeting_date'] = dates.strftime('%Y-%m-%d')
        panel['cutoff'] = [(day - pd.Timedelta(days=1)).strftime('%Y-%m-%dT16:00:00-04:00') for day in dates]
        panel['label'] = ['cut', 'hold', 'hike', 'hold'] * 11
        meetings = [{'date': row['meeting_date'], 'label': row['label']} for _, row in panel.iterrows()]
        with patch('ratewatch.forecast.fit_model', wraps=fit_model) as fitted:
            predictions = evaluate_meetings(panel, meetings)
        self.assertEqual(len(predictions), 4)
        for call, prediction in zip(fitted.call_args_list, predictions):
            train = call.args[0]
            self.assertTrue((train['meeting_date'] < prediction['cutoff'][:10]).all())
            self.assertAlmostEqual(sum(prediction['model']), 1)
        changed = panel.copy()
        changed.loc[43, features] = 1000
        changed.loc[43, 'label'] = 'cut'
        self.assertEqual(predictions[:-1], evaluate_meetings(changed, meetings)[:-1])

    def test_no_upcoming_meeting(self):
        data = {'cutoff': datetime(2026, 9, 13, tzinfo=timezone.utc),
                'meetings': [{'date': '2026-01-28', 'label': 'hold'}]}
        self.assertEqual(forecast_next(data), {'status': 'no upcoming verified meeting'})

    def test_issue_retains_parameters_and_scores_named_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'meetings').mkdir()
            (folder / 'protocol.json').write_text(json.dumps({'version': 'test'}))
            cutoff = '2026-09-13T12:00:00+00:00'
            forecast = {'meeting_date': '2026-10-28', 'information_cutoff': cutoff,
                        'probabilities': {'cut': 0.2, 'hold': 0.7, 'hike': 0.1},
                        'model_parameters': {'classes': ['cut', 'hike', 'hold'], 'coef': [[1], [2], [3]]}}
            data = {'folder': folder, 'offline': True}
            with self.assertRaises(ValueError):
                issue_forecast(data, {}, forecast)
            data['offline'] = False
            evaluation = {'panel': [{'meeting_date': '2026-01-28', 'label': 'hold'}]}
            with patch('ratewatch.audit.datetime') as clock:
                clock.now.return_value = datetime(2026, 9, 13, 12, 1, tzinfo=timezone.utc)
                clock.fromisoformat.side_effect = datetime.fromisoformat
                issue_forecast(data, evaluation, forecast)
                with self.assertRaises(ValueError):
                    issue_forecast(data, evaluation, forecast)
            row = read_log(folder / 'audit/forecasts.jsonl')[0]
            snapshot = json.loads((folder / 'audit/snapshots' / (row['snapshot_hash'] + '.json')).read_text())
            self.assertEqual(snapshot['model']['coef'], [[1], [2], [3]])
            self.assertEqual(snapshot['training_panel'], evaluation['panel'])
            self.assertIn('ratewatch/forecast.py', snapshot['code_hashes'])
            result = score_log(folder / 'audit', [{'date': '2026-10-28', 'label': 'hold'}])
            self.assertAlmostEqual(result['early']['scores']['log_loss'], -np.log(0.7))

    def test_compressed_cache_and_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            url = 'https://www.bankofcanada.ca/example/'
            cached = Download(Path(directory) / 'pages')
            cached.session = Mock()
            cached.session.get.return_value.text = 'first'
            with patch('ratewatch.sources.time.sleep'):
                self.assertEqual(cached.get(url)['body'], 'first')
                self.assertEqual(cached.get(url)['body'], 'first')
                cached.session.get.assert_called_once()
                refreshed = Download(Path(directory) / 'pages', refresh=True)
                refreshed.session = Mock()
                refreshed.session.get.return_value.text = 'second'
                self.assertEqual(refreshed.get(url)['body'], 'second')
                self.assertEqual(cached.get(url)['body'], 'second')
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['pages.json.gz'])

    def test_refresh_keeps_release_calendar_provenance(self):
        schedule_url = 'https://www.bankofcanada.ca/schedule-2026/'
        with tempfile.TemporaryDirectory() as directory:
            with patch('ratewatch.sources.archive_links', return_value={schedule_url: '2026 schedule'}), \
                 patch('ratewatch.sources.parse_schedule', return_value=['2026-01-28']), \
                 patch('ratewatch.sources.Download.get', return_value={
                     'body': '[]', 'retrieved_at': '2026-09-13T12:00:00+00:00'}), \
                 patch('ratewatch.sources.announcement_label', return_value=('hold', 'Bank holds rate')):
                meetings, releases = refresh_sources(directory, start=2026, end=2026)
            self.assertEqual(meetings[0]['label'], 'hold')
            self.assertEqual(releases, [])
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            self.assertEqual(manifest['release_calendar'], release_calendar_url)


if __name__ == '__main__':
    unittest.main()
