"""Descriptive turnover analytics: how many leave, why, and where."""

from __future__ import annotations

import pandas as pd

REASONS = ["resignation", "retirement", "layoff"]


def yearly_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Exit rate per year and reason, as a share of everyone employed during the year."""
    counts = panel.pivot_table(
        index="year", columns="exit_reason", values="employee_id", aggfunc="count", fill_value=0
    )
    counts = counts.reindex(columns=["none", *REASONS], fill_value=0)
    table = pd.DataFrame({"headcount": counts.sum(axis=1)})
    for reason in REASONS:
        table[f"{reason}s"] = counts[reason]
        table[f"{reason}_rate_pct"] = 100 * counts[reason] / table["headcount"]
    table["total_exit_rate_pct"] = 100 * counts[REASONS].sum(axis=1) / table["headcount"]
    return table


def rates_by(panel: pd.DataFrame, column: str, min_person_years: int = 200) -> pd.DataFrame:
    """Resignation rate per group, with a Wilson 95 % interval so small groups are not over-read."""
    grouped = panel.groupby(column)["resigned"].agg(person_years="size", resignations="sum")
    grouped = grouped[grouped["person_years"] >= min_person_years]
    n, k = grouped["person_years"], grouped["resignations"]
    p, z = k / n, 1.96
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / (1 + z**2 / n)
    grouped["rate_pct"] = 100 * p
    grouped["ci_low_pct"] = 100 * (centre - half)
    grouped["ci_high_pct"] = 100 * (centre + half)
    return grouped.sort_values("rate_pct", ascending=False)


def band(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=edges, labels=labels, right=False).astype(str)


AGE_EDGES, AGE_LABELS = [0, 25, 35, 45, 55, 200], ["under 25", "25-34", "35-44", "45-54", "55+"]
TENURE_EDGES, TENURE_LABELS = (
    [0, 1, 3, 5, 10, 100],
    ["under 1 yr", "1-2 yrs", "3-4 yrs", "5-9 yrs", "10+ yrs"],
)


def add_bands(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.assign(
        age_band=band(panel["age"], AGE_EDGES, AGE_LABELS),
        tenure_band=band(panel["tenure"], TENURE_EDGES, TENURE_LABELS),
    )
