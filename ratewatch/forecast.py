import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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
            'log_loss': float(-np.log(np.clip(probability[np.arange(len(actual)), encoded], 1e-12, 1)).mean()),
            'brier': float(np.mean(np.sum((probability - truth)**2, axis=1)))}


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


def summarize(rows):
    names = ['model', 'frequency', 'transition', 'always_hold', 'persistence']
    metrics = {name: score([r['actual'] for r in rows], [r[name] for r in rows]) for name in names}
    return {'n': len(rows), 'start': rows[0]['meeting_date'], 'end': rows[-1]['meeting_date'],
            'scores': metrics}


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
    model_accuracy = evaluation['scores']['model']['accuracy']
    baseline_accuracy = evaluation['scores']['persistence']['accuracy']
    comparison = 'exceeded' if model_accuracy > baseline_accuracy else 'did not exceed'
    lines += ['', f'Model accuracy {comparison} the repeat-previous-decision baseline on this sample.',
              'Lower log loss and Brier score are better; deterministic baselines are mainly accuracy comparisons.', '',
              'Historical values may be revised. Same-day releases are excluded. '
              'Only scheduled decisions are tested, and six indicators omit other policy drivers.',
              f"{len(evaluation['omitted'])} meetings lacked usable inputs."]
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')
    return folder / 'report.md'
