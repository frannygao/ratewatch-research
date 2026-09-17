"""Turn the raw series into one row per meeting.

Each row holds the six indicators the model reads, computed from only the data
that was already public before that meeting's cutoff. The cutoff is 16:00
Toronto on the previous weekday, which is when the Bank's decision is
effectively locked in.

Nothing here knows about the model. This module answers one question: what did
the world look like on the eve of each decision?
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import (FEATURES, INFLATION_TARGET, NORMAL_MONTHS, STALE_DAYS,
                     TORONTO, TREND_MONTHS)

Series = dict[str, pd.Series]
Release = dict[str, str]
Meeting = dict[str, Any]

__all__ = ['FEATURES', 'build_panel', 'cutoff_for', 'feature_row',
           'latest_release', 'next_meeting', 'previous_label']


def cutoff_for(meeting_date: str) -> datetime:
    """16:00 Toronto on the weekday before the meeting.

    A full Bank of Canada holiday calendar was tried here and moved this date on
    none of the 120 meetings in the record, so the simple version is kept.
    """
    day = date.fromisoformat(meeting_date) - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return datetime.combine(day, time(16), TORONTO)


def latest_release(releases: list[Release], series: str, cutoff: datetime) -> Release:
    """The most recent figure for a series that was public before the cutoff."""
    available = [r for r in releases
                 if r['series'] == series and r['available_after'] <= cutoff.date().isoformat()]
    if not available:
        raise ValueError(f'No published {series} figure before the cutoff.')
    return max(available, key=lambda r: (r['reference'], r['published']))


def feature_row(series: Series, releases: list[Release],
                cutoff: datetime) -> tuple[dict[str, float], dict[str, Any]]:
    """The six indicators as of one cutoff, plus a record of what was used.

    Raises ValueError when the inputs are missing or too old to trust, which is
    how meetings without usable history get dropped from the panel.
    """
    if cutoff.tzinfo is None:
        raise ValueError('A timezone-aware cutoff is required.')
    local = cutoff.astimezone(TORONTO)
    last_day = pd.Timestamp(local.date()) - pd.Timedelta(days=1)

    policy = series['policy'].loc[:last_day]
    if policy.empty or (last_day - policy.index[-1]).days > 7:
        raise ValueError('Policy rate observations are missing or stale.')

    cpi_release = latest_release(releases, 'cpi', local)
    labour_release = latest_release(releases, 'unemployment', local)
    for release in (cpi_release, labour_release):
        if (local.date() - date.fromisoformat(release['published'])).days > STALE_DAYS:
            raise ValueError('Latest published figure is too old to use.')

    cpi = series['cpi'].resample('MS').last()
    unemployment = series['unemployment'].resample('MS').last()
    inflation = cpi.pct_change(12, fill_method=None) * 100
    cpi_month = pd.Timestamp(cpi_release['reference'])
    labour_month = pd.Timestamp(labour_release['reference'])

    rate = float(policy.iloc[-1])
    rate_earlier = float(policy.loc[:last_day - pd.DateOffset(months=TREND_MONTHS)].iloc[-1])
    normal = unemployment.rolling(NORMAL_MONTHS).mean()

    values = {
        'inflation_gap': float(inflation.loc[cpi_month]) - INFLATION_TARGET,
        'inflation_trend': float(inflation.diff(TREND_MONTHS).loc[cpi_month]),
        'unemployment_change': float(unemployment.diff(TREND_MONTHS).loc[labour_month]),
        'unemployment_gap': float((unemployment - normal).loc[labour_month]),
        'real_rate': rate - float(inflation.loc[cpi_month]),
        'policy_momentum': rate - rate_earlier,
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError('Insufficient feature history.')

    used = {'cutoff': local.isoformat(), 'policy_rate': rate,
            'policy_date': str(policy.index[-1].date()),
            'cpi': cpi_release, 'unemployment': labour_release}
    return values, used


def build_panel(series: Series, releases: list[Release],
                meetings: list[Meeting]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """One row per decided meeting, in date order.

    `previous_label` is the decision at the meeting before. It is not a model
    input; it is what lets the report separate meetings that continued a course
    from meetings that changed one.

    Returns the panel and a list of meetings that had to be dropped, so the
    report can say how many and why.
    """
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    previous: str | None = None

    for meeting in meetings:
        if meeting['label'] is None:
            continue
        cutoff = cutoff_for(meeting['date'])
        try:
            values, used = feature_row(series, releases, cutoff)
        except (ValueError, KeyError, IndexError) as error:
            dropped.append({'meeting': meeting['date'], 'reason': str(error)})
            continue
        rows.append({**values, 'meeting_date': meeting['date'], 'cutoff': cutoff.isoformat(),
                     'label': meeting['label'], 'previous_label': previous, 'used': used})
        previous = meeting['label']

    if not rows:
        raise ValueError('No meetings have usable feature history.')
    return pd.DataFrame(rows), dropped


def previous_label(row: pd.Series | dict[str, Any]) -> str | None:
    """The decision at the meeting before this one, or None for the first row.

    pandas stores a missing string as NaN, so this turns it back into None.
    """
    value = row['previous_label']
    return value if isinstance(value, str) else None


def next_meeting(meetings: list[Meeting], now: datetime) -> Meeting | None:
    """The soonest scheduled meeting that has not been decided yet."""
    upcoming = [m for m in meetings if m['label'] is None and cutoff_for(m['date']) >= now]
    return min(upcoming, key=lambda m: m['date']) if upcoming else None
