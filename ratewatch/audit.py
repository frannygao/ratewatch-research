import fcntl
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .forecast import deadline, label_order, score, training


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def stage_for(meeting, issued):
    cutoff = deadline(meeting)
    if issued.tzinfo is None or issued > cutoff:
        raise ValueError('Forecast must be issued before the deadline with a timezone.')
    return 'deadline' if issued >= cutoff - timedelta(minutes=60) else 'early'


def read_log(path):
    path = Path(path)
    rows = []
    previous = None
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        row = json.loads(line)
        signature = row.pop('hash')
        if row['previous_hash'] != previous or digest(row) != signature:
            raise ValueError('Forecast log integrity check failed.')
        row['hash'] = signature
        rows.append(row)
        previous = signature
    return rows


def append_forecast(folder, record, snapshot):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    issued = datetime.fromisoformat(record['issued_at'])
    if record['stage'] != stage_for(record['meeting_date'], issued):
        raise ValueError('Incorrect forecast stage.')
    if datetime.fromisoformat(record['information_cutoff']) > issued:
        raise ValueError('Information cutoff cannot be in the future.')
    path = folder / 'forecasts.jsonl'
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_log(path)
        key = ('meeting_date', 'protocol_version', 'stage')
        if any(all(old[k] == record[k] for k in key) for old in rows):
            raise ValueError('This meeting, protocol and stage already has a forecast.')
        snapshots = folder / 'snapshots'
        snapshots.mkdir(exist_ok=True)
        signature = digest(snapshot)
        saved = snapshots / (signature + '.json')
        if not saved.exists():
            with saved.open('xb') as output:
                output.write(encoded(snapshot))
        elif hashlib.sha256(saved.read_bytes()).hexdigest() != signature:
            raise ValueError('Snapshot integrity check failed.')
        entry = dict(record, snapshot_hash=signature, previous_hash=rows[-1]['hash'] if rows else None)
        entry['hash'] = digest(entry)
        with path.open('a') as output:
            output.write(encoded(entry).decode() + '\n')
            output.flush()
            os.fsync(output.fileno())
    return entry


def score_log(folder, meetings):
    folder = Path(folder)
    rows = read_log(folder / 'forecasts.jsonl')
    actual = {m['date']: m['label'] for m in meetings if m['label']}
    result = {}
    for row in rows:
        saved = folder / 'snapshots' / (row['snapshot_hash'] + '.json')
        if hashlib.sha256(saved.read_bytes()).hexdigest() != row['snapshot_hash']:
            raise ValueError('Snapshot integrity check failed.')
    versions = {r['protocol_version'] for r in rows}
    if len(versions) > 1:
        raise ValueError('Score protocol versions separately; do not pool changed models.')
    for stage in ['deadline', 'early']:
        scored = [r for r in rows if r['stage'] == stage and r['meeting_date'] in actual]
        result[stage] = {'issued': sum(r['stage'] == stage for r in rows), 'scored': len(scored),
                         'scores': score([actual[r['meeting_date']] for r in scored],
                                         [probability_values(r['probabilities']) for r in scored]) if scored else None}
        result[stage]['baseline_scores'] = {}
        for name in ['frequency', 'transition', 'always_hold', 'persistence']:
            if scored and all(name in r.get('baselines', {}) for r in scored):
                result[stage]['baseline_scores'][name] = score(
                    [actual[r['meeting_date']] for r in scored],
                    [r['baselines'][name] for r in scored])
    return result


def probability_values(probabilities):
    return [probabilities[label] for label in label_order] if isinstance(probabilities, dict) else probabilities


def issue_forecast(data, evaluation, forecast):
    import pandas as pd
    import sklearn

    if data['offline']:
        raise ValueError('Issuing requires fresh downloads.')
    if 'probabilities' not in forecast:
        raise ValueError('No verified future meeting is available.')
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((data['folder'] / 'protocol.json').read_text())
    issued = datetime.now(timezone.utc)
    record = dict(forecast, issued_at=issued.isoformat(), stage=stage_for(forecast['meeting_date'], issued),
                  protocol_version=protocol['version'], status='issued')
    train = training(pd.DataFrame(evaluation['panel']), forecast['information_cutoff'])
    snapshot = {'protocol': protocol, 'forecast': record,
                'training_panel': json.loads(train.to_json(orient='records')),
                'model': dict(forecast['model_parameters'], sklearn_version=sklearn.__version__),
                'code_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [root / 'run.py', *sorted((root / 'ratewatch').glob('*.py'))]},
                'sources': {str(p.relative_to(data['folder'])): json.loads(p.read_text())
                            for folder in [data['folder'], data['folder'] / 'meetings'] for p in folder.glob('*.json')}}
    append_forecast(data['folder'] / 'audit', record, snapshot)
    return record
