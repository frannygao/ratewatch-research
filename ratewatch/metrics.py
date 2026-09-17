"""Measure how good the forecasts actually are.

Four scores, and three things a single accuracy number cannot tell you:

  scores()          accuracy, balanced accuracy, log loss, Brier score
  by_meeting_type() the same, split by whether the Bank changed course
  mcnemar()         whether a gap between two methods is bigger than chance
  bootstrap_ci()    how wide the uncertainty around a score really is
  brier_skill()     one number for "better than this reference, or not"

Why the split matters. Roughly three quarters of meetings repeat the previous
decision, so any method that just says "same as last time" scores well overall
while being useless. Splitting the meetings shows whether a model can do the
only hard thing: call a change before it happens.

Why log loss is reported as n/a for some methods. A method that forecasts 0% or
100% takes an unbounded penalty when it is wrong, so its log loss is decided by
the tiny floor used to keep the logarithm finite rather than by the forecast.
Reporting a number there would invite a comparison that means nothing. For the
same reason, note that the Brier score of a 0/100 forecast is just twice its
error rate, so it carries no information beyond accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Any

import numpy as np

from .config import (BOOTSTRAP_DRAWS, CONFIDENCE, LABELS, LOG_FLOOR,
                     MEETING_TYPES, REFERENCE)

Prediction = dict[str, Any]

__all__ = ['Evaluation', 'binomial_p', 'bootstrap_ci', 'brier_skill',
           'by_meeting_type', 'correct_mask', 'difference_ci', 'evaluate',
           'mcnemar', 'meeting_type', 'scores']


@dataclass
class Evaluation:
    """Every measurement of one backtest."""

    n: int
    first: str
    last: str
    reference: str
    scores: dict[str, dict[str, Any]]
    by_meeting_type: dict[str, dict[str, Any]]
    skill: dict[str, float | None]
    significance: dict[str, dict[str, Any]]
    accuracy_ci: dict[str, dict[str, Any]]
    difference_ci: dict[str, dict[str, Any]]
    predictions: list[Prediction]
    dropped: list[dict[str, str]]

    def accuracy(self, method: str = 'model') -> float:
        return self.scores[method]['accuracy']

    def moves_called(self, method: str = 'model') -> tuple[int, int] | None:
        """Hits and total on the meetings where the Bank started or resumed moving."""
        group = self.by_meeting_type[method].get('new move')
        return (group['correct'], group['n']) if group else None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _encode(actual: list[str]) -> np.ndarray:
    return np.array([LABELS.index(a) for a in actual])


def _is_deterministic(probabilities: np.ndarray) -> bool:
    return bool(np.all((probabilities == 0) | (probabilities == 1)))


def scores(actual: list[str], probabilities: Any) -> dict[str, Any]:
    """The four headline scores for one set of forecasts.

    accuracy          share of meetings where the highest probability was right
    balanced_accuracy mean of the per-class recalls, so rare cuts and hikes
                      count as much as common holds
    log_loss          mean of -log(probability given to what happened); rewards
                      being confident only when justified. Lower is better.
    brier             mean squared error across all three probabilities, summed
                      per meeting, so it runs 0 (perfect) to 2 (confidently
                      wrong). Lower is better.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    truth = _encode(actual)
    predicted = probabilities.argmax(axis=1)

    matrix = np.zeros((3, 3), int)
    for row, column in zip(truth, predicted):
        matrix[row][column] += 1
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(3), where=support > 0)

    assigned = probabilities[np.arange(len(truth)), truth]
    onehot = np.eye(3)[truth]
    return {
        'n': int(len(truth)),
        'accuracy': float((predicted == truth).mean()),
        'balanced_accuracy': float(recall[support > 0].mean()),
        'log_loss': None if _is_deterministic(probabilities)
                    else float(-np.log(np.clip(assigned, LOG_FLOOR, 1)).mean()),
        'brier': float(((probabilities - onehot) ** 2).sum(axis=1).mean()),
        'recall': {label: (float(recall[i]) if support[i] else None)
                   for i, label in enumerate(LABELS)},
        'confusion': matrix.tolist(),
    }


def meeting_type(previous_label: Any, label: str) -> str:
    """Classify a meeting by whether the Bank changed course.

    Anything that is not a recognised decision, including a missing first-row
    value, counts as unchanged: there is no earlier decision to have changed.
    """
    if previous_label not in LABELS or previous_label == label:
        return 'unchanged'
    return 'returned to hold' if label == 'hold' else 'new move'


def by_meeting_type(predictions: list[Prediction], method: str) -> dict[str, dict[str, Any]]:
    """Accuracy for one method, split by whether the Bank changed course."""
    result = {}
    for kind in MEETING_TYPES:
        group = [p for p in predictions
                 if meeting_type(p['previous_label'], p['actual']) == kind]
        if not group:
            continue
        probabilities = np.asarray([p[method] for p in group], dtype=float)
        correct = probabilities.argmax(axis=1) == _encode([p['actual'] for p in group])
        result[kind] = {'n': len(group), 'correct': int(correct.sum()),
                        'accuracy': float(correct.mean()),
                        'meetings': [p['meeting_date'] for p in group]}
    return result


def correct_mask(predictions: list[Prediction], method: str) -> np.ndarray:
    probabilities = np.asarray([p[method] for p in predictions], dtype=float)
    return probabilities.argmax(axis=1) == _encode([p['actual'] for p in predictions])


def binomial_p(successes: int, trials: int) -> float:
    """Two-sided exact binomial test against a fair coin."""
    if trials == 0:
        return 1.0
    smaller = min(successes, trials - successes)
    tail = sum(comb(trials, i) for i in range(smaller + 1))
    return min(1.0, 2 * tail / 2 ** trials)


def mcnemar(predictions: list[Prediction], method: str, reference: str) -> dict[str, Any]:
    """Is one method's accuracy edge over another bigger than chance?

    Only the meetings where the two disagree carry information. If the edge is
    real, the wins should not split evenly, so the disagreements are tested
    against a fair coin. A large p-value means the sample cannot tell them apart.
    """
    a, b = correct_mask(predictions, method), correct_mask(predictions, reference)
    wins, losses = int((a & ~b).sum()), int((~a & b).sum())
    return {'method': method, 'reference': reference,
            'method_accuracy': float(a.mean()), 'reference_accuracy': float(b.mean()),
            'wins': wins, 'losses': losses, 'p_value': binomial_p(wins, wins + losses)}


def bootstrap_ci(predictions: list[Prediction], method: str, metric: str = 'accuracy',
                 level: float = CONFIDENCE, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = 0) -> dict[str, Any]:
    """A confidence interval for one score, by resampling meetings.

    The meetings are drawn with replacement many times and the score recomputed
    each time; the interval is the middle `level` of those values. It answers
    "how much would this number move on a different run of history?"
    """
    rng = np.random.default_rng(seed)
    actual = [p['actual'] for p in predictions]
    probabilities = np.asarray([p[method] for p in predictions], dtype=float)
    tail = (1 - level) / 2 * 100

    values = [scores([actual[i] for i in index], probabilities[index])[metric]
              for index in rng.integers(0, len(actual), (draws, len(actual)))]
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return {'metric': metric, 'point': scores(actual, probabilities)[metric],
            'low': float(np.percentile(values, tail)),
            'high': float(np.percentile(values, 100 - tail)), 'level': level}


def difference_ci(predictions: list[Prediction], method: str, reference: str,
                  level: float = CONFIDENCE, draws: int = BOOTSTRAP_DRAWS,
                  seed: int = 0) -> dict[str, Any]:
    """A confidence interval for the accuracy GAP between two methods.

    Both methods forecast the same meetings, so this resamples meetings and
    recomputes the difference within each resample, keeping the pairing intact.
    That is the honest way to ask whether one is better: comparing two separate
    intervals invites the overlap fallacy, since two intervals can overlap while
    the difference is real, and vice versa. An interval straddling zero means
    this record cannot say which method is better.
    """
    rng = np.random.default_rng(seed)
    a, b = correct_mask(predictions, method), correct_mask(predictions, reference)
    index = rng.integers(0, len(a), (draws, len(a)))
    gaps = a[index].mean(axis=1) - b[index].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return {'point': float(a.mean() - b.mean()),
            'low': float(np.percentile(gaps, tail)),
            'high': float(np.percentile(gaps, 100 - tail)),
            'method_ahead_share': float((gaps > 0).mean()), 'level': level}


def brier_skill(predictions: list[Prediction], method: str, reference: str) -> float | None:
    """How much of the reference method's error this method removes.

    1.0 is perfect, 0.0 is no better than the reference, negative is worse. This
    is the single number to look at when asking whether the model earns its keep.
    Undefined, and returned as None, against a reference that never errs.
    """
    actual = [p['actual'] for p in predictions]
    model = scores(actual, [p[method] for p in predictions])['brier']
    base = scores(actual, [p[reference] for p in predictions])['brier']
    return None if base == 0 else float(1 - model / base)


def evaluate(predictions: list[Prediction], methods: list[str],
             dropped: list[dict[str, str]] | None = None,
             reference: str = REFERENCE) -> Evaluation:
    """Everything above, for every method, in one object."""
    actual = [p['actual'] for p in predictions]
    return Evaluation(
        n=len(predictions),
        first=predictions[0]['meeting_date'],
        last=predictions[-1]['meeting_date'],
        reference=reference,
        scores={m: scores(actual, [p[m] for p in predictions]) for m in methods},
        by_meeting_type={m: by_meeting_type(predictions, m) for m in methods},
        skill={m: brier_skill(predictions, m, reference) for m in methods},
        significance={m: mcnemar(predictions, m, reference)
                      for m in methods if m != reference},
        accuracy_ci={m: bootstrap_ci(predictions, m) for m in methods},
        difference_ci={m: difference_ci(predictions, m, reference)
                       for m in methods if m != reference},
        predictions=predictions,
        dropped=dropped or [],
    )
