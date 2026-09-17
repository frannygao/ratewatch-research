"""Keep a dated record of forecasts made before the fact, and score them later.

The backtest can only ever be a reconstruction. It uses today's revised CPI and
unemployment figures, so it shows what the model would conclude from the data as
it now reads, not as it read at the time. Release dates are handled correctly;
revisions are not, because the revised numbers are all that exist.

The only clean score is a forecast written down before the decision and checked
afterwards. This module is that record: `add()` appends one forecast, `score()`
grades every logged forecast whose meeting has since been decided.

Forecasts are grouped by protocol version and scored separately, so a change to
the model never gets to borrow the track record of the old one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import metrics
from .config import FORECAST_LOG
from .features import Meeting, cutoff_for
from .model import Forecast

LOG = 'forecasts.jsonl'

__all__ = ['add', 'read', 'score']


def read(folder: Path | str = FORECAST_LOG) -> list[dict[str, Any]]:
    path = Path(folder) / LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add(folder: Path | str, forecast: Forecast, protocol_version: str) -> dict[str, Any]:
    """Append one forecast to the log.

    Refuses a forecast issued after the decision was locked in, and refuses a
    second forecast for the same meeting under the same protocol, so the log
    cannot be quietly improved after the fact.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    issued = datetime.now(timezone.utc)
    if issued > cutoff_for(forecast.meeting_date):
        raise ValueError("That meeting's cutoff has already passed.")

    for existing in read(folder):
        if (existing['meeting_date'], existing['protocol_version']) == \
           (forecast.meeting_date, protocol_version):
            raise ValueError(f'{forecast.meeting_date} already has a '
                             f'{protocol_version} forecast.')

    entry = {'meeting_date': forecast.meeting_date,
             'protocol_version': protocol_version,
             'issued_at': issued.isoformat(),
             'information_cutoff': forecast.information_cutoff,
             'probabilities': forecast.probabilities,
             'baselines': forecast.baselines,
             'features': forecast.features}
    with (folder / LOG).open('a') as output:
        output.write(json.dumps(entry) + '\n')
    return entry


def score(folder: Path | str, meetings: list[Meeting]) -> dict[str, dict[str, Any]]:
    """Grade logged forecasts against what the Bank actually did.

    Returns one block per protocol version, each saying how many forecasts were
    issued, how many can be graded yet, and how they scored.
    """
    decided = {m['date']: m['label'] for m in meetings if m['label']}
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in read(folder):
        by_version[entry['protocol_version']].append(entry)

    result = {}
    for version, entries in sorted(by_version.items()):
        gradable = [e for e in entries if e['meeting_date'] in decided]
        block: dict[str, Any] = {'issued': len(entries), 'graded': len(gradable),
                                 'scores': None}
        if gradable:
            actual = [decided[e['meeting_date']] for e in gradable]
            block['scores'] = {
                'model': metrics.scores(
                    actual, [list(e['probabilities'].values()) for e in gradable]),
                **{name: metrics.scores(
                    actual, [list(e['baselines'][name].values()) for e in gradable])
                   for name in gradable[0].get('baselines', {})},
            }
        result[version] = block
    return result
