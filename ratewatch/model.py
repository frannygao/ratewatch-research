"""Fit the model, replay it over history, and forecast the next meeting.

The model is deliberately small: standardise the six indicators, then fit a
multinomial logistic regression over cut / hold / hike. The specification is
fixed in data/protocol.json and is not tuned against the backtest, because a
model chosen by its score on these meetings would have a meaningless score.

The backtest is a walk-forward replay. At each meeting the scaler and the
regression are refitted from scratch on earlier meetings only, so nothing from
the future can leak backwards. There is no random train/test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, LABELS, MAX_ITER, MIN_TRAIN, PENALTY, TORONTO
from .features import (Meeting, Release, Series, cutoff_for, feature_row,
                       next_meeting, previous_label)

Prediction = dict[str, Any]

__all__ = ['Forecast', 'backtest', 'baselines', 'fit', 'forecast', 'probabilities']


@dataclass
class Forecast:
    """The model's view of one upcoming meeting."""

    meeting_date: str
    meeting_cutoff: str
    information_cutoff: str
    probabilities: dict[str, float]
    features: dict[str, float]
    used: dict[str, Any]
    baselines: dict[str, dict[str, float]]
    trained_on: int
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def leading(self) -> str:
        """The most likely decision."""
        return max(self.probabilities, key=self.probabilities.__getitem__)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit(train: pd.DataFrame) -> Pipeline:
    """Standardise, then fit multinomial logistic regression."""
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=PENALTY, max_iter=MAX_ITER))
    model.fit(train[FEATURES], train['label'])
    return model


def probabilities(model: Pipeline, row: pd.DataFrame | dict[str, float]) -> list[float]:
    """Predicted probabilities as a list in LABELS order."""
    frame = pd.DataFrame([row]) if isinstance(row, dict) else row
    predicted = model.predict_proba(frame[FEATURES])[0]
    order = list(model.classes_)
    return [float(predicted[order.index(label)]) for label in LABELS]


def baselines(history: list[str]) -> dict[str, list[float]]:
    """Four reference forecasts, each using only decisions before this meeting.

    frequency    how often each decision has happened, lightly smoothed
    transition   what followed this same decision in the past
    always_hold  always hold
    persistence  repeat the previous decision; the one to beat, because it is
                 right whenever the Bank stays the course
    """
    counts = np.array([history.count(label) for label in LABELS], dtype=float)
    previous = history[-1]
    followed = [after for before, after in zip(history[:-1], history[1:]) if before == previous]
    followed_counts = np.array([followed.count(label) for label in LABELS], dtype=float)
    return {
        'frequency': ((counts + 1) / (len(history) + 3)).tolist(),
        'transition': ((followed_counts + 1) / (len(followed) + 3)).tolist(),
        'always_hold': [0.0, 1.0, 0.0],
        'persistence': [float(label == previous) for label in LABELS],
    }


def earlier_than(panel: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Panel rows for meetings that had already been decided before a cutoff."""
    day = pd.Timestamp(cutoff).tz_convert(TORONTO).date().isoformat()
    return panel[panel['meeting_date'] < day]


def ready(train: pd.DataFrame) -> bool:
    """Enough history to fit? Needs MIN_TRAIN meetings covering all three decisions."""
    return len(train) >= MIN_TRAIN and train['label'].nunique() == len(LABELS)


def backtest(panel: pd.DataFrame) -> list[Prediction]:
    """Replay the model over history, one meeting at a time.

    A meeting is only scored once there is enough earlier history, so the early
    years act as a warm-up rather than being predicted from nothing.
    """
    predictions: list[Prediction] = []
    for _, row in panel.iterrows():
        train = earlier_than(panel, row['cutoff'])
        if not ready(train):
            continue
        predictions.append({
            'meeting_date': row['meeting_date'],
            'cutoff': row['cutoff'],
            'actual': row['label'],
            'previous_label': previous_label(row),
            'trained_on': len(train),
            'model': probabilities(fit(train), row.to_frame().T),
            **baselines(train['label'].tolist()),
        })
    if not predictions:
        raise ValueError('Not enough decided meetings to score a backtest.')
    return predictions


def forecast(panel: pd.DataFrame, series: Series, releases: list[Release],
             meetings: list[Meeting], now: datetime) -> Forecast | None:
    """Probabilities for the next scheduled meeting, using data available now."""
    meeting = next_meeting(meetings, now)
    if meeting is None:
        return None

    values, used = feature_row(series, releases, now)
    train = earlier_than(panel, now.isoformat())
    if not ready(train):
        raise ValueError(f'Need at least {MIN_TRAIN} earlier meetings '
                         'covering all three decisions.')

    model = fit(train)
    scaler, regression = model.steps[0][1], model.steps[1][1]
    history = train['label'].tolist()
    return Forecast(
        meeting_date=meeting['date'],
        meeting_cutoff=cutoff_for(meeting['date']).isoformat(),
        information_cutoff=now.isoformat(),
        probabilities=dict(zip(LABELS, probabilities(model, values))),
        features=values,
        used=used,
        baselines={name: dict(zip(LABELS, probs))
                   for name, probs in baselines(history).items()},
        trained_on=len(train),
        parameters={'classes': regression.classes_.tolist(),
                    'coefficients': regression.coef_.tolist(),
                    'intercepts': regression.intercept_.tolist(),
                    'feature_means': scaler.mean_.tolist(),
                    'feature_scales': scaler.scale_.tolist()},
    )
