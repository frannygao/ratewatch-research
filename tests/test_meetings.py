import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ratewatch.audit import append_forecast, read_log, score_log, stage_for
from ratewatch.forecast import deadline, released_reference, feature_row
from ratewatch.forecast import baselines, training
from ratewatch.sources import announcement_label, parse_releases, parse_schedule


class MeetingTests(unittest.TestCase):
    def test_announcement_not_effective_date(self):
        html = '<main><h1>Bank of Canada lowers policy rate by 25 basis points</h1></main>'
        self.assertEqual(announcement_label(html)[0], 'cut')
        html = '<main><h1>Bank of Canada will hold current level of policy rate</h1></main>'
        self.assertEqual(announcement_label(html)[0], 'hold')
        with self.assertRaises(ValueError):
            announcement_label('<main><h1>Policy announcement</h1></main>')

    def test_schedule_paragraph(self):
        html = '<main><p>January 25<br>March 8<br>April 12<br>June 7<br>July 12<br>September 6<br>October 25<br>December 6</p></main>'
        self.assertEqual(len(parse_schedule(html, 2023)), 8)
        with self.assertRaises(ValueError):
            parse_schedule('<main><p>January 25</p></main>', 2023)

    def test_releases_require_publication(self):
        row = {'title': 'Consumer Price Index', 'description': 'July 2026',
               'date': '2026-08-17 00:00:01', 'url': '/daily-quotidien/260817/example.htm'}
        self.assertEqual(parse_releases([dict(row, url='')], date(2026, 9, 1)), [])
        self.assertEqual(parse_releases([row], date(2026, 8, 16)), [])
        releases = parse_releases([row], date(2026, 9, 1))
        with self.assertRaises(ValueError):
            released_reference(releases, 'cpi', datetime.fromisoformat('2026-08-17T16:00:00-04:00'))
        self.assertEqual(released_reference(releases, 'cpi', datetime.fromisoformat('2026-08-18T16:00:00-04:00'))['reference'], '2026-07-01')

    def test_unreleased_values_cannot_change_features(self):
        import numpy as np
        months = pd.date_range('2020-01-01', '2026-12-01', freq='MS')
        days = pd.date_range('2020-01-01', '2026-12-31', freq='D')
        raw = {'policy': pd.Series(3.0, index=days),
               'cpi': pd.Series(100 * 1.002 ** np.arange(len(months)), index=months),
               'unemployment': pd.Series(6.0, index=months)}
        releases = [{'series': series, 'reference': '2026-07-01',
                     'published_date': '2026-08-17', 'available_after': '2026-08-18'}
                    for series in ['cpi', 'unemployment']]
        cutoff = datetime.fromisoformat('2026-09-01T16:00:00-04:00')
        before, _ = feature_row(raw, releases, cutoff)
        for name in ['cpi', 'unemployment']:
            raw[name].loc['2026-08-01':] *= 100
        raw['policy'].loc['2026-09-01':] *= 100
        after, audit = feature_row(raw, releases, cutoff)
        self.assertEqual(before, after)
        self.assertEqual(audit['policy_date'], '2026-08-31')

    def test_deadline_holidays_and_dst(self):
        self.assertEqual(deadline('2026-10-28').isoformat(), '2026-10-27T16:00:00-04:00')
        self.assertEqual(deadline('2026-09-08').date(), date(2026, 9, 4))
        self.assertEqual(deadline('2026-12-29').date(), date(2026, 12, 24))
        self.assertEqual(deadline('2026-01-28').isoformat(), '2026-01-27T16:00:00-05:00')

    def test_training_excludes_unannounced_labels(self):
        panel = pd.DataFrame({'meeting_date': ['2026-01-28', '2026-03-18', '2026-04-29']})
        self.assertEqual(training(panel, '2026-03-17T16:00:00-04:00')['meeting_date'].tolist(), ['2026-01-28'])

    def test_smoothed_baselines(self):
        result = baselines([{'label': x} for x in ['hold', 'cut', 'cut']])
        self.assertAlmostEqual(sum(result['transition']), 1)
        self.assertTrue(all(x > 0 for x in result['transition']))
        self.assertEqual(result['persistence'], [1, 0, 0])

    def test_issue_window(self):
        self.assertEqual(stage_for('2026-10-28', datetime.fromisoformat('2026-10-27T14:59:00-04:00')), 'early')
        self.assertEqual(stage_for('2026-10-28', datetime.fromisoformat('2026-10-27T15:00:00-04:00')), 'deadline')
        with self.assertRaises(ValueError):
            stage_for('2026-10-28', datetime.fromisoformat('2026-10-27T16:01:00-04:00'))

    def test_log_duplicate_and_tampering(self):
        record = {'meeting_date': '2026-10-28', 'issued_at': '2026-09-13T12:00:00+00:00',
                  'information_cutoff': '2026-09-13T11:59:00+00:00', 'stage': 'early',
                  'protocol_version': 'test', 'probabilities': [0.2, 0.7, 0.1]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entry = append_forecast(folder, record, {'inputs': [1, 2]})
            self.assertEqual(len(read_log(folder / 'forecasts.jsonl')), 1)
            with self.assertRaises(ValueError):
                append_forecast(folder, record, {})
            self.assertEqual(score_log(folder, [])['early']['scored'], 0)
            self.assertEqual(score_log(folder, [{'date': '2026-10-28', 'label': 'hold'}])['early']['scored'], 1)
            snapshot = folder / 'snapshots' / (entry['snapshot_hash'] + '.json')
            snapshot.write_text('{}')
            with self.assertRaises(ValueError):
                score_log(folder, [])
            path = folder / 'forecasts.jsonl'
            changed = json.loads(path.read_text())
            changed['probabilities'] = [0, 1, 0]
            path.write_text(json.dumps(changed) + '\n')
            with self.assertRaises(ValueError):
                read_log(path)


if __name__ == '__main__':
    unittest.main()
