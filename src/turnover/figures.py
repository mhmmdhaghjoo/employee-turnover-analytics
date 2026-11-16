"""Charts for the report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6a65", "#e1e0d9", "#fcfcfb"
STYLE: dict[str, Any] = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
}  # fmt: skip


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def exits_by_year(rates: pd.DataFrame, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        bottom = np.zeros(len(rates))
        for reason, color in (("resignation", BLUE), ("retirement", ORANGE), ("layoff", AQUA)):
            values = rates[f"{reason}_rate_pct"].to_numpy()
            ax.bar(np.arange(len(rates)), values, bottom=bottom, color=color, label=reason.capitalize(),
                   width=0.7, edgecolor=SURFACE, linewidth=1.5)  # fmt: skip
            bottom += values
        ax.set_xticks(np.arange(len(rates)), rates.index.astype(str))
        for x, total in enumerate(bottom):
            ax.text(x, total + 0.08, f"{total:.1f}", ha="center", fontsize=9, color=INK)
        ax.set_title("Exits per year as a share of everyone employed that year (%)")
        ax.legend(frameon=False, loc="upper left", ncols=3)
        ax.set_ylim(0, bottom.max() * 1.25)
        _save(fig, path)


def rate_bars(table: pd.DataFrame, title: str, path: Path, order: list[str] | None = None) -> None:
    data = table.reindex(order) if order else table.sort_values("rate_pct")
    data = data.dropna(subset=["rate_pct"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(data) + 1.4))
        errors = [data["rate_pct"] - data["ci_low_pct"], data["ci_high_pct"] - data["rate_pct"]]
        ax.barh(data.index.astype(str), data["rate_pct"], xerr=errors, color=BLUE, ecolor=MUTED, height=0.6)
        for y, (rate, high) in enumerate(zip(data["rate_pct"], data["ci_high_pct"], strict=True)):
            ax.text(high + 0.05, y, f"{rate:.1f}%", va="center", fontsize=9, color=INK)
        ax.tick_params(axis="y", colors=INK)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(title)
        ax.set_xlabel("Resignations per 100 employee-years (bars: 95 % interval)")
        _save(fig, path)


def survival_curves(tables: dict[str, pd.DataFrame], path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        for (label, table), color in zip(tables.items(), (BLUE, ORANGE, AQUA), strict=False):
            x = np.concatenate([[0], table.index.to_numpy() + 1])
            survival = np.concatenate([[1.0], table["survival"].to_numpy()])
            se = np.concatenate([[0.0], table["survival_se"].to_numpy()])
            ax.step(x, 100 * survival, where="post", color=color, linewidth=2, label=label)
            ax.fill_between(
                x,
                100 * (survival - 1.96 * se),
                100 * (survival + 1.96 * se),
                step="post",
                color=color,
                alpha=0.15,
            )
            ax.text(
                x[-1] + 0.15,
                100 * survival[-1],
                f"{100 * survival[-1]:.0f}%",
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.set_xlabel("Years since hire")
        ax.set_title("Share of a hiring cohort that has not resigned (%)")
        ax.legend(frameon=False, loc="lower left")
        _save(fig, path)


def capture_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot([0, 100], [0, 100], color=GRID, linestyle="--", linewidth=1)
        ax.text(72, 66, "no model", color=MUTED, fontsize=9, rotation=38)
        share = 100 * np.arange(1, len(y) + 1) / len(y)
        for (name, score), color in zip(scores.items(), (MUTED, ORANGE, BLUE), strict=False):
            order = np.argsort(-score, kind="stable")
            ax.plot(
                share,
                100 * np.cumsum(y[order]) / y.sum(),
                color=color,
                linewidth=2,
                label=name.replace("_", " "),
            )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of employees reviewed, highest risk first (%)")
        ax.set_ylabel("Share of resignations found (%)")
        ax.set_title("How many leavers does a risk list find?")
        ax.grid(axis="x", visible=True)
        ax.legend(frameon=False, loc="lower right")
        _save(fig, path)
