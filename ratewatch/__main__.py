"""RateWatch: probability of a Bank of Canada cut, hold or hike.

    python -m ratewatch            backtest, forecast, figures, report
    python -m ratewatch fetch      refresh data/ from the official sources
    python -m ratewatch record     log the current forecast, to grade later

The pipeline runs in six steps, one module each:

    sources   read the saved CPI, unemployment and policy-rate series, the
              meeting calendar, and the day each figure was published
    features  build one row per meeting from the figures that were public
              before that meeting's cutoff
    model     refit at every meeting on earlier meetings only and predict the
              one in front of it; then forecast the next undecided meeting
    metrics   score those predictions, split by whether the Bank changed
              course, with a significance test and confidence intervals
    viz       draw the four figures
    report    write results/report.md and the full numbers as JSON

Downloading is a separate command on purpose. Once data/ is populated every
other step reads from disk, so a run is reproducible and needs no network.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from . import features, metrics, model, record, report, sources, viz
from .config import FORECAST_LOG, METHODS, PROTOCOL

COMMANDS = ('run', 'fetch', 'record')


def analyse() -> tuple[metrics.Evaluation, model.Forecast | None, list]:
    """Saved data to a finished evaluation, printing each step as it goes."""
    series = sources.load_series()
    meetings, releases = sources.load_meetings()
    print(f'  loaded {len(meetings)} meetings, {len(releases)} published figures')

    panel, dropped = features.build_panel(series, releases, meetings)
    print(f'  panel: {len(panel)} usable meetings, {len(dropped)} dropped')

    predictions = model.backtest(panel)
    print(f'  walk-forward: {len(predictions)} meetings scored')

    evaluation = metrics.evaluate(predictions, METHODS, dropped=dropped)
    current = model.forecast(panel, series, releases, meetings,
                             datetime.now(timezone.utc))

    figures = viz.draw_all(evaluation, panel, series)
    print(f'  figures: {len(figures)} written')
    return evaluation, current, figures, meetings


def summarise(evaluation: metrics.Evaluation, forecast: model.Forecast | None) -> None:
    score = evaluation.scores['model']
    print(f'\nTested {evaluation.n} meetings '
          f'({evaluation.first} to {evaluation.last}).')
    print(f"  accuracy {score['accuracy']:.1%}"
          f"  balanced {score['balanced_accuracy']:.1%}"
          f"  log loss {score['log_loss']:.3f}"
          f"  skill vs {evaluation.reference} {evaluation.skill['model']:+.2f}")
    called = evaluation.moves_called()
    if called:
        print(f'  called {called[0]} of {called[1]} changes of course.')
    if forecast:
        shown = ', '.join(f'{k} {v:.1%}' for k, v in forecast.probabilities.items())
        print(f'Next meeting {forecast.meeting_date}: {shown}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m ratewatch',
                                     description=__doc__.splitlines()[0])
    parser.add_argument('command', nargs='?', default='run', choices=COMMANDS,
                        help='fetch: refresh data. run: backtest and forecast. '
                             'record: log the current forecast.')
    command = parser.parse_args(argv).command

    if command == 'fetch':
        sources.fetch()
        return

    print('RateWatch · reading saved data')
    evaluation, current, figures, meetings = analyse()

    if command == 'record':
        if current is None:
            raise SystemExit('No undecided meeting to forecast.')
        version = json.loads(PROTOCOL.read_text())['version']
        try:
            entry = record.add(FORECAST_LOG, current, version)
        except ValueError as refused:
            raise SystemExit(f'Not recorded: {refused}')
        print(f"  recorded {entry['meeting_date']} under {version}")

    prospective = record.score(FORECAST_LOG, meetings)
    path = report.write(evaluation, current, prospective, figures)
    summarise(evaluation, current)
    print(f'Report: {path}')


if __name__ == '__main__':
    main()
