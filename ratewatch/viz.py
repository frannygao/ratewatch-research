"""Two figures, written to results/figures/.

  macro_trends  why does it fail?  the inputs against what the Bank did
  calibration   anything left?     whether the probabilities can be trusted

Only two, because a figure has to carry something a table cannot. The
meeting-type split, the confusion matrix and the per-meeting probabilities on
the meetings that changed course are all tables in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

from .config import COLOURS, FIGURE_DPI, FIGURES, LABELS  # noqa: E402
from .metrics import Evaluation  # noqa: E402

__all__ = ['calibration', 'draw_all', 'macro_trends']


def _save(fig: plt.Figure, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def macro_trends(panel: pd.DataFrame, series: dict[str, pd.Series], folder: Path) -> Path:
    """The model's inputs, against what the Bank actually did."""
    cpi = series['cpi'].resample('MS').last()
    inflation = cpi.pct_change(12, fill_method=None) * 100
    unemployment = series['unemployment'].resample('MS').last()
    start = pd.Timestamp(panel['meeting_date'].min())
    inflation, unemployment = inflation.loc[start:], unemployment.loc[start:]
    policy = series['policy'].loc[start:]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(inflation.index, inflation.values, color=COLOURS['model'], lw=1.6)
    axes[0].axhline(2.0, color=COLOURS['rule'], ls='--', lw=1, label='2% target')
    axes[0].set_ylabel('CPI inflation (%)')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    axes[0].set_title('Model inputs, and the decisions they were meant to explain', fontsize=11)

    axes[1].plot(unemployment.index, unemployment.values, color=COLOURS['reference'], lw=1.6)
    axes[1].set_ylabel('Unemployment (%)')

    axes[2].step(policy.index, policy.values, where='post', color='#333', lw=1.4)
    axes[2].set_ylabel('Policy rate (%)')
    axes[2].set_xlabel('')

    moves = panel[panel['label'] != 'hold']
    for _, row in moves.iterrows():
        axes[2].axvline(pd.Timestamp(row['meeting_date']), color=COLOURS[row['label']],
                        alpha=0.45, lw=1.1)
    handles = [plt.Line2D([], [], color=COLOURS[k], lw=1.4, label=k) for k in ('cut', 'hike')]
    axes[2].legend(handles=handles, frameon=False, fontsize=8, loc='upper left')

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _save(fig, 'macro_trends.png', folder)


def calibration(evaluation: Evaluation, folder: Path, bins: int = 6) -> Path:
    """Are the probabilities honest, even when the call is wrong?

    Every forecast contributes three points, one per decision. If the model says
    20% and that thing happens a fifth of the time, the curve sits on the
    diagonal. This is what a Brier skill score above zero looks like.
    """
    probabilities = np.asarray([p['model'] for p in evaluation.predictions], dtype=float)
    happened = np.zeros_like(probabilities)
    for i, p in enumerate(evaluation.predictions):
        happened[i, LABELS.index(p['actual'])] = 1

    flat_p, flat_y = probabilities.ravel(), happened.ravel()
    edges = np.linspace(0, 1, bins + 1)
    slot = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)

    centres, observed, weights = [], [], []
    for b in range(bins):
        mask = slot == b
        if mask.sum() < 3:
            continue
        centres.append(flat_p[mask].mean())
        observed.append(flat_y[mask].mean())
        weights.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls='--', color=COLOURS['rule'], lw=1, label='perfectly calibrated')
    ax.plot(centres, observed, 'o-', color=COLOURS['model'], lw=1.8, ms=7, label='model')
    for x, y, w in zip(centres, observed, weights):
        ax.annotate(f'n={w}', (x, y), textcoords='offset points', xytext=(7, -11),
                    fontsize=7.5, color='#555')

    skill = evaluation.skill.get('model')
    subtitle = f'Brier skill vs {evaluation.reference}: {skill:+.2f}' if skill is not None else ''
    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'Are the probabilities honest?\n{subtitle}', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return _save(fig, 'calibration.png', folder)


def draw_all(evaluation: Evaluation, panel: pd.DataFrame,
             series: dict[str, pd.Series], folder: Path = FIGURES) -> list[Path]:
    """Every figure, in the order they are worth reading."""
    return [macro_trends(panel, series, folder),
            calibration(evaluation, folder)]
