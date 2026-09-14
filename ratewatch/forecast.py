import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


toronto = ZoneInfo('America/Toronto')
features = ['inflation_gap', 'inflation_trend', 'unemployment_change',
           'unemployment_gap', 'real_rate', 'policy_momentum']
label_order = ['cut', 'hold', 'hike']


def bank_holidays(year):
    fixed = [date(year, 1, 1), date(year, 7, 1), date(year, 11, 11),
             date(year, 12, 25), date(year, 12, 26)]
    if year >= 2021:
        fixed.append(date(year, 9, 30))
    observed = set(fixed)
    for day in sorted(fixed):
        if day.weekday() >= 5:
            shifted = day
            while shifted.weekday() >= 5 or shifted in observed:
                shifted += timedelta(days=1)
            observed.add(shifted)
    observed |= {easter(year) - timedelta(days=2), easter(year) + timedelta(days=1)}
    for month, occurrence in [(2, 3), (8, 1), (9, 1), (10, 2)]:
        day = date(year, month, 1)
        day += timedelta(days=(0 - day.weekday()) % 7 + 7 * (occurrence - 1))
        observed.add(day)
    day = date(year, 5, 24)
    observed.add(day - timedelta(days=(day.weekday() - 0) % 7))
    return observed


def deadline(meeting_date):
    day = date.fromisoformat(meeting_date) - timedelta(days=1)
    while day.weekday() >= 5 or day in bank_holidays(day.year):
        day -= timedelta(days=1)
    return datetime.combine(day, time(16), toronto)


def released_reference(releases, series, cutoff):
    eligible = [r for r in releases if r['series'] == series and r['available_after'] <= cutoff.date().isoformat()]
    if not eligible:
        raise ValueError('No verified release available for ' + series)
    return max(eligible, key=lambda r: (r['reference'], r['published_date']))


def feature_row(raw, releases, cutoff):
    if cutoff.tzinfo is None:
        raise ValueError('A timezone-aware cutoff is required.')
    local = cutoff.astimezone(toronto)
    last_day = pd.Timestamp(local.date()) - pd.Timedelta(days=1)
    policy = raw['policy'].loc[:last_day]
    if policy.empty or (last_day - policy.index[-1]).days > 7:
        raise ValueError('Policy observations are missing or stale.')
    cpi_release = released_reference(releases, 'cpi', local)
    labour_release = released_reference(releases, 'unemployment', local)
    for release in [cpi_release, labour_release]:
        if (local.date() - date.fromisoformat(release['published_date'])).days > 65:
            raise ValueError('Economic release is stale.')
    cpi = raw['cpi'].resample('MS').last()
    unemployment = raw['unemployment'].resample('MS').last()
    cpi_date = pd.Timestamp(cpi_release['reference'])
    labour_date = pd.Timestamp(labour_release['reference'])
    inflation = cpi.pct_change(12, fill_method=None) * 100
    rate = float(policy.iloc[-1])
    earlier = policy.loc[:last_day - pd.DateOffset(months=3)]
    values = {'inflation_gap': float(inflation.loc[cpi_date] - 2),
              'inflation_trend': float(inflation.diff(3).loc[cpi_date]),
              'unemployment_change': float(unemployment.diff(3).loc[labour_date]),
              'unemployment_gap': float((unemployment - unemployment.rolling(36).mean()).loc[labour_date]),
              'real_rate': rate - float(inflation.loc[cpi_date]),
              'policy_momentum': rate - float(earlier.iloc[-1])}
    if not np.isfinite(list(values.values())).all():
        raise ValueError('Insufficient feature history.')
    audit = {'cutoff': local.isoformat(), 'policy_date': str(policy.index[-1].date()), 'policy_rate': rate,
             'cpi': cpi_release, 'unemployment': labour_release,
             'vintage': 'current numerical vintage; historical values may be revised'}
    return values, audit


def build_meeting_panel(raw, releases, meetings):
    rows, omitted = [], []
    for meeting in meetings:
        if meeting['label'] is None:
            continue
        cutoff = deadline(meeting['date'])
        try:
            values, audit = feature_row(raw, releases, cutoff)
            rows.append(dict(values, meeting_date=meeting['date'], cutoff=cutoff.isoformat(),
                             label=meeting['label'], audit=audit, announcement_url=meeting['announcement_url']))
        except (ValueError, KeyError, IndexError) as error:
            omitted.append({'meeting': meeting['date'], 'reason': str(error)})
    if not rows:
        raise ValueError('No meetings have sufficient release-aware feature history.')
    return pd.DataFrame(rows).sort_values('meeting_date').reset_index(drop=True), omitted


def fit_model(train):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    model.fit(train[features], train['label'])
    return model


def predict(model, row):
    p = model.predict_proba(row[features])[0]
    return [float(p[list(model.classes_).index(label)]) for label in label_order]


def predict_probabilities(model, row):
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    return dict(zip(label_order, predict(model, row)))


def training(panel, cutoff):
    stamp = datetime.fromisoformat(cutoff)
    if stamp.tzinfo is None:
        raise ValueError('A timezone-aware cutoff is required.')
    day = stamp.astimezone(toronto).date().isoformat()
    return panel[panel['meeting_date'] < day]


def baselines(prior_meetings):
    labels = [m['label'] for m in prior_meetings]
    counts = np.array([labels.count(label) for label in label_order], float)
    frequencies = (counts + 1) / (len(labels) + 3)
    previous = labels[-1]
    transitions = [b for a, b in zip(labels[:-1], labels[1:]) if a == previous]
    transition_counts = np.array([transitions.count(label) for label in label_order], float)
    return {'frequency': frequencies.tolist(),
            'transition': ((transition_counts + 1) / (len(transitions) + 3)).tolist(),
            'always_hold': [0, 1, 0], 'persistence': [int(label == previous) for label in label_order]}


def score(actual, probability):
    actual = np.asarray(actual)
    probability = np.asarray(probability)
    predicted = np.array(label_order)[probability.argmax(axis=1)]
    encoded = np.array([label_order.index(y) for y in actual])
    matrix = confusion_matrix(actual, predicted, labels=label_order)
    support = matrix.sum(axis=1)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(3), where=support != 0)
    truth = np.eye(3)[encoded]
    return {'accuracy': float(accuracy_score(actual, predicted)),
            'balanced_accuracy': float(recalls[support > 0].mean()),
            'macro_f1': float(f1_score(actual, predicted, labels=label_order, average='macro', zero_division=0)),
            'log_loss': float(-np.log(np.clip(probability[np.arange(len(actual)), encoded], 1e-12, 1)).mean()),
            'brier': float(np.mean(np.sum((probability - truth)**2, axis=1))),
            'recall': dict(zip(label_order, recalls.tolist())),
            'support': dict(zip(label_order, support.tolist())),
            'confusion_matrix': matrix.tolist()}


def evaluate_meetings(panel, meetings):
    predictions = []
    for _, row in panel.iterrows():
        train = training(panel, row['cutoff'])
        if len(train) < 40 or train['label'].nunique() != 3:
            continue
        prior = [m for m in meetings if m['label'] and m['date'] < row['cutoff'][:10]]
        predictions.append({'meeting_date': row['meeting_date'], 'cutoff': row['cutoff'],
                            'actual': row['label'], 'model': predict(fit_model(train), row.to_frame().T),
                            'training_meetings': len(train), 'last_training_meeting': train['meeting_date'].max(),
                            **baselines(prior)})
    if not predictions:
        raise ValueError('Insufficient meetings to evaluate all three classes.')
    return predictions


def loss_interval(rows, baseline):
    loss = lambda r, key: -np.log(max(r[key][label_order.index(r['actual'])], 1e-12))
    improvement = np.array([loss(r, baseline) - loss(r, 'model') for r in rows])
    rng = np.random.default_rng(42)
    n = len(rows)
    samples = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / 4)))
        index = np.concatenate([(s + np.arange(4)) % n for s in starts])[:n]
        samples.append(improvement[index].mean())
    return {'mean_log_loss_improvement': float(improvement.mean()),
            'interval_95': np.quantile(samples, [0.025, 0.975]).tolist(), 'block_meetings': 4}


def summarize(rows):
    names = ['model', 'frequency', 'transition', 'always_hold', 'persistence']
    metrics = {name: score([r['actual'] for r in rows], [r[name] for r in rows]) for name in names}
    calibration = []
    for label in label_order:
        for low, high in [(0, 1/3), (1/3, 2/3), (2/3, 1.000001)]:
            subset = [r for r in rows if low <= r['model'][label_order.index(label)] < high]
            if subset:
                calibration.append({'class': label, 'bin': [low, min(high, 1)], 'n': len(subset),
                                    'mean_probability': float(np.mean([r['model'][label_order.index(label)] for r in subset])),
                                    'observed_fraction': float(np.mean([r['actual'] == label for r in subset]))})
    return {'n': len(rows), 'start': rows[0]['meeting_date'], 'end': rows[-1]['meeting_date'],
            'primary_metric': 'multiclass log loss', 'scores': metrics,
            'vs_frequency': loss_interval(rows, 'frequency'), 'vs_transition': loss_interval(rows, 'transition'),
            'calibration': calibration,
            'market_comparison': 'unavailable: no meeting-specific timestamped OIS data'}


def evaluate(data):
    panel, omitted = build_meeting_panel(data['raw'], data['releases'], data['meetings'])
    rows = evaluate_meetings(panel, data['meetings'])
    return {**summarize(rows), 'predictions': rows, 'omitted': omitted,
            'panel': json.loads(panel.to_json(orient='records'))}


def forecast_next(data):
    now = data['cutoff']
    upcoming = [m for m in data['meetings'] if m['label'] is None and deadline(m['date']) >= now]
    if not upcoming:
        return {'status': 'no upcoming verified meeting'}
    meeting = min(upcoming, key=lambda row: row['date'])
    values, audit = feature_row(data['raw'], data['releases'], now)
    panel, _ = build_meeting_panel(data['raw'], data['releases'], data['meetings'])
    train = training(panel, now.isoformat())
    if len(train) < 40 or train['label'].nunique() != 3:
        raise ValueError('At least 40 earlier meetings and all three classes are required.')
    model = fit_model(train)
    prior = [m for m in data['meetings'] if m['label'] and m['date'] < now.astimezone(toronto).date().isoformat()]
    scaler, regression = model.steps[0][1], model.steps[1][1]
    return {'meeting_date': meeting['date'], 'information_cutoff': now.isoformat(),
            'deadline': deadline(meeting['date']).isoformat(), 'class_order': label_order,
            'probabilities': predict_probabilities(model, values), 'features': values,
            'audit': audit, 'baselines': baselines(prior), 'training_meetings': len(train),
            'model_parameters': {'classes': model.classes_.tolist(), 'coef': regression.coef_.tolist(),
                                 'intercept': regression.intercept_.tolist(),
                                 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()},
            'status': 'saved snapshot preview' if data['offline'] else 'preview'}


def save_results(evaluation, forecast, prospective, folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parents[1] / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    evaluation = dict(evaluation, prospective=prospective)
    for name, value in [('evaluation', evaluation), ('forecast', forecast)]:
        (folder / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    lines = ['# RateWatch', '', '## Next BoC meeting', '']
    if 'probabilities' in forecast:
        lines += [f"{forecast['meeting_date']}. {forecast['status'].capitalize()}. Cutoff: {forecast['information_cutoff']}.", '',
                  '| Decision | Probability |', '| --- | ---: |']
        lines += [f"| {label.capitalize()} | {forecast['probabilities'][label]:.1%} |" for label in label_order]
    else:
        lines += [forecast['status']]
    lines += ['', '## Historical evaluation', '',
              f"{evaluation['n']} scheduled meetings, {evaluation['start']} to {evaluation['end']}.", '',
              '| Method | Accuracy | Balanced accuracy | Log loss | Brier score |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, values in evaluation['scores'].items():
        lines.append(f"| {name} | {values['accuracy']:.1%} | {values['balanced_accuracy']:.1%} | {values['log_loss']:.3f} | {values['brier']:.3f} |")
    lines += ['', 'Deterministic baselines are chiefly accuracy comparisons. Lower log loss and Brier score are better.',
              '', '## Interpretation', '']
    if 'probabilities' in forecast:
        winner = max(forecast['probabilities'], key=forecast['probabilities'].get)
        lines += [f"The model assigns the highest probability to a {winner}. These estimates use six macro indicators; policy-context text does not affect them."]
    lines += ['Historical results test whether the macro model adds value; they do not establish a forecasting advantage.',
              '', '## Prospective record', '']
    for stage, values in prospective.items():
        lines += [f"{stage.capitalize()}: {values['issued']} issued, {values['scored']} scored."]
    lines += ['', 'Only explicitly issued audit records count as prospective forecasts. Early and deadline forecasts are scored separately.',
              '', '## Limitations', '',
              '- Historical inputs use current numerical vintages and may contain revisions.',
              '- Historical results are reconstructions. Same-day releases are excluded.',
              '- Only scheduled decisions are targets; six features omit other policy drivers.',
              '- No timestamped meeting-specific market benchmark is available.',
              '- Local hashes detect edits but cannot prove issue time or prevent wholesale rewriting.',
              f"- {len(evaluation['omitted'])} meetings lacked usable inputs; details are in evaluation.json."]
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')
    return folder / 'report.md'
