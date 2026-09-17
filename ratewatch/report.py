"""Render results/report.md.

The report leads with the question that decides whether this project is worth
anything: can the model call a change of course before it happens? Overall
accuracy comes second, because on a record where most meetings repeat the
previous decision, a high overall score mostly measures how often nothing
happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import LABELS, MEETING_TYPES, METHODS, RESULTS
from .metrics import Evaluation, meeting_type
from .model import Forecast

__all__ = ['write']

Rows = list[list[Any]]


def percent(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1%}'


def number(value: float | None, places: int = 3) -> str:
    return 'n/a' if value is None else f'{value:.{places}f}'


def table(header: list[str], rows: Rows) -> list[str]:
    rule = ['---'] + ['---:'] * (len(header) - 1)
    return ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(rule) + ' |'] + \
           ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]


def next_meeting_section(forecast: Forecast | None) -> list[str]:
    if forecast is None:
        return ['No scheduled meeting is currently undecided.']
    when = forecast.meeting_cutoff[:16].replace('T', ' ')
    known = forecast.information_cutoff[:16].replace('T', ' ')
    cpi, jobs = forecast.used['cpi'], forecast.used['unemployment']
    return [f'**{forecast.meeting_date}**, decided by {when} Toronto.',
            f'Forecast from data available at {known} UTC.', ''] + \
        table(['Decision', 'Probability'],
              [[label.capitalize(), percent(forecast.probabilities[label])]
               for label in LABELS]) + \
        ['', f'Fitted on {forecast.trained_on} earlier meetings. Latest inputs: '
             f"CPI for {cpi['reference'][:7]} (published {cpi['published']}), "
             f"unemployment for {jobs['reference'][:7]} (published {jobs['published']}), "
             f"policy rate {forecast.used['policy_rate']}%."]


def change_section(evaluation: Evaluation, methods: list[str]) -> list[str]:
    splits = evaluation.by_meeting_type
    kinds = [k for k in MEETING_TYPES if k in splits['model']]
    rows: Rows = []
    for kind in kinds:
        row: list[Any] = [kind.capitalize(), splits['model'][kind]['n']]
        row += [f"{splits[m][kind]['correct']}/{splits[m][kind]['n']}"
                f" ({splits[m][kind]['accuracy']:.0%})" for m in methods]
        rows.append(row)

    lines = table(['Meeting type', 'n'] + [m.replace('_', ' ') for m in methods], rows)
    lines += [''] + [f'- **{kind.capitalize()}** {MEETING_TYPES[kind]}' for kind in kinds]

    called = evaluation.moves_called()
    if called:
        moves = splits['model']['new move']
        lines += ['', f'The {moves["n"]} meetings where the Bank started or resumed '
                      'moving the rate: ' + ', '.join(moves['meetings']) + '.',
                  '', f'The model called {called[0]} of them.']
    return lines


def scores_section(evaluation: Evaluation, methods: list[str]) -> list[str]:
    rows: Rows = []
    for method in methods:
        score = evaluation.scores[method]
        interval = evaluation.accuracy_ci[method]
        rows.append([method.replace('_', ' '),
                     f"{percent(score['accuracy'])} "
                     f"[{percent(interval['low'])}, {percent(interval['high'])}]",
                     percent(score['balanced_accuracy']),
                     number(score['log_loss']), number(score['brier']),
                     number(evaluation.skill[method], 2)])
    return table(['Method', 'Accuracy (95% CI)', 'Balanced accuracy', 'Log loss',
                  'Brier', f'Skill vs {evaluation.reference}'], rows)


def moves_section(evaluation: Evaluation) -> list[str]:
    """What the model said on each meeting where the Bank changed course.

    Only a handful of these exist, so they are listed rather than summarised: a
    rate over eight meetings cannot distinguish a narrow miss from a confident
    one, and that distinction is the whole question.
    """
    moves = [p for p in evaluation.predictions
             if meeting_type(p['previous_label'], p['actual']) == 'new move']
    if not moves:
        return []
    rows: Rows = []
    for row in moves:
        given = row['model'][LABELS.index(row['actual'])]
        rows.append([row['meeting_date'], row['actual'],
                     percent(given), percent(row['model'][LABELS.index('hold')])])
    even = 1 / len(LABELS)
    return table(['Meeting', 'What the Bank did', 'Probability it gave that',
                  'Probability it gave hold'], rows) + \
        ['', f'An even guess would have given {even:.0%} to each option. No baseline '
             'called any of these either.']


def confusion_section(evaluation: Evaluation) -> list[str]:
    matrix = evaluation.scores['model']['confusion']
    rows: Rows = [[f'**{label.capitalize()}**', *matrix[i]]
                  for i, label in enumerate(LABELS)]
    return table(['Actual \\ predicted'] + [l.capitalize() for l in LABELS], rows)


def significance_section(evaluation: Evaluation) -> list[str]:
    rows: Rows = []
    for method, test in evaluation.significance.items():
        gap = evaluation.difference_ci[method]
        rows.append([method.replace('_', ' '), percent(test['method_accuracy']),
                     percent(test['reference_accuracy']),
                     f"{gap['point']:+.1%} [{gap['low']:+.1%}, {gap['high']:+.1%}]",
                     f"{test['wins']}/{test['losses']}", number(test['p_value'], 3)])
    lines = table(['Method', 'Accuracy', evaluation.reference.capitalize(),
                   'Difference (95% CI)', 'Wins/losses', 'p'], rows)
    return lines + [
        '', 'The difference is the statistic that matters, and it is measured paired: '
            'both methods forecast the same meetings, so the interval comes from '
            'resampling meetings and recomputing the gap. An interval straddling zero '
            'means this record cannot say which method is better. Comparing each '
            "method's own interval instead would be misleading, because two intervals "
            'can overlap while the difference is real.',
        '', 'Wins and losses count only the meetings where the two disagreed.']


def prospective_section(prospective: dict[str, dict[str, Any]]) -> list[str]:
    if not prospective:
        return ['No forecasts have been recorded yet. Run `python -m ratewatch record` '
                'before a meeting to start the record.']
    lines: list[str] = []
    for version, block in prospective.items():
        lines.append(f"**{version}**: {block['issued']} issued, {block['graded']} graded.")
        if block['scores']:
            lines += [''] + table(
                ['Method', 'Accuracy', 'Log loss', 'Brier'],
                [[name.replace('_', ' '), percent(s['accuracy']),
                  number(s['log_loss']), number(s['brier'])]
                 for name, s in block['scores'].items()])
        lines.append('')
    return lines


def figures_section(figures: list[Path], folder: Path) -> list[str]:
    if not figures:
        return []
    lines = ['', '## Figures', '']
    for path in figures:
        relative = path.relative_to(folder) if path.is_relative_to(folder) else path.name
        lines += [f'![{path.stem.replace("_", " ")}]({relative})', '']
    return lines


def write(evaluation: Evaluation, forecast: Forecast | None,
          prospective: dict[str, dict[str, Any]], figures: list[Path] | None = None,
          methods: list[str] = METHODS, folder: Path = RESULTS) -> Path:
    """Write results/report.md plus the full numbers as JSON."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    figures = figures or []

    lines = ['# RateWatch', '',
             'Probability of a Bank of Canada cut, hold or hike at the next scheduled '
             'meeting,',
             'from official CPI, unemployment and policy-rate data published before the '
             'decision.',
             '', '## Next meeting', '']
    lines += next_meeting_section(forecast)

    lines += ['', '## Can it call a change of course?', '',
              f'{evaluation.n} meetings tested, {evaluation.first} to {evaluation.last}.',
              'Most meetings repeat the previous decision, so overall accuracy mostly '
              'measures how',
              'often nothing happened. This table separates the meetings where something '
              'did.', '']
    lines += change_section(evaluation, methods)

    lines += ['', '## Overall scores', '']
    lines += scores_section(evaluation, methods)
    lines += ['', 'Lower log loss and Brier score are better. Skill is the share of the '
                  f"{evaluation.reference} baseline's squared error removed: 0 means no "
                  'better, negative means worse.',
              '', 'Log loss is left blank for methods that forecast only 0% or 100%. '
                  'Their penalty for being wrong depends on the floor used to keep the '
                  'logarithm finite, not on the forecast, so the number would not be '
                  'comparable.']

    lines += ['', 'What it said on each meeting that changed course:', '']
    lines += moves_section(evaluation)

    lines += ['', 'Where the walk-forward predictions landed:', '']
    lines += confusion_section(evaluation)

    lines += ['', '## Is any difference real?', '']
    lines += significance_section(evaluation)

    lines += ['', '## Forecasts on the record', '',
              'The sections above are a reconstruction: they use the current, revised CPI',
              'and unemployment figures. Only forecasts written down before a decision are',
              'a clean test.', '']
    lines += prospective_section(prospective)
    lines += figures_section(figures, folder)

    lines += ['## Limits', '',
              f'- {len(evaluation.dropped)} meetings were dropped for want of usable '
              'published inputs.',
              '- Historical figures carry later revisions; release dates are '
              'point-in-time, values are not.',
              '- A figure published on the day of a cutoff is treated as unavailable, '
              'because the',
              '  release calendar records dates but not times.',
              '- Only scheduled decisions are tested, and six indicators leave out much '
              'of what the',
              '  Bank discusses: core inflation measures, the output gap, oil, and US '
              'policy.',
              '- This forecasts direction, not the size of a move.']

    (folder / 'report.md').write_text('\n'.join(lines) + '\n')
    (folder / 'evaluation.json').write_text(json.dumps(
        {**evaluation.as_dict(), 'prospective': prospective}, indent=2) + '\n')
    (folder / 'forecast.json').write_text(json.dumps(
        forecast.as_dict() if forecast else None, indent=2) + '\n')
    return folder / 'report.md'
