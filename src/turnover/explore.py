"""Group-level views for the people analytics app: cohort survival, stratified rates, fairness.

Every function returns aggregates over groups of employees, never a score or
a label for an individual. Small groups are suppressed rather than shown,
because a rate computed on a handful of people is both unreliable and close
to identifying someone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from turnover.audit import fairness_audit
from turnover.data import build_panel, load_records
from turnover.demo import make_raw
from turnover.describe import add_bands
from turnover.survival import life_table

Z = 1.959964


def load_panel(data_path: Path) -> tuple[pd.DataFrame, bool]:
    """The analysis panel and whether it is the synthetic demo.

    Uses the real file when it exists; otherwise the synthetic records from
    ``turnover.demo``, so the app still runs without the dataset.
    """
    if data_path.exists():
        return add_bands(build_panel(load_records(data_path))), False
    demo_path = data_path.parent / "_demo_records.csv"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    make_raw().to_csv(demo_path, index=False)
    return add_bands(build_panel(load_records(demo_path))), True


def survival_curves(
    panel: pd.DataFrame, column: str | None = None, max_tenure: int = 10, min_at_risk: int = 30
) -> pd.DataFrame:
    """Survival (still employed, not resigned) by completed tenure year, per group.

    Adds a 95 % pointwise interval from Greenwood's standard error, clipped to
    [0, 1]. Tenure years where fewer than ``min_at_risk`` employees were at
    risk are dropped: the curve stops where the data gets thin.
    """
    groups = [("All employees", panel)] if column is None else list(panel.groupby(column))
    frames = []
    for name, group in groups:
        table = life_table(group, max_tenure).reset_index()
        table = table[table["at_risk"] >= min_at_risk]
        if table.empty:
            continue
        table["low"] = (table["survival"] - Z * table["survival_se"]).clip(0, 1)
        table["high"] = (table["survival"] + Z * table["survival_se"]).clip(0, 1)
        table.insert(0, "group", str(name))
        frames.append(table)
    columns = ["group", "tenure_year", "at_risk", "resignations", "hazard", "survival", "low", "high"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def stratified_rates(
    panel: pd.DataFrame, rows: str, columns: str, min_person_years: int = 100
) -> pd.DataFrame:
    """Resignation rate (%) for every combination of two attributes.

    Cells with fewer than ``min_person_years`` employee-years are set to NaN
    (suppressed). Returns a long table with the counts so the caller can show
    both the rate and how much data stands behind it.
    """
    grouped = panel.groupby([rows, columns], observed=True)["resigned"].agg(
        person_years="size", resignations="sum"
    )
    grouped = grouped.reset_index()
    grouped["rate_pct"] = 100 * grouped["resignations"] / grouped["person_years"]
    grouped["suppressed"] = grouped["person_years"] < min_person_years
    grouped.loc[grouped["suppressed"], "rate_pct"] = np.nan
    return grouped


def group_fairness(
    test: pd.DataFrame, scores: np.ndarray, column: str, flag_fraction: float, min_group: int = 100
) -> pd.DataFrame:
    """Fairness of a 'flag the top X %' rule, per group, with the two ratios that summarise it.

    ``flag_rate_ratio`` compares how often groups are flagged; ``false_flag_ratio``
    compares how often people who do *not* resign are flagged anyway. Both are
    lowest group / highest group, so 1.0 is parity. Groups smaller than
    ``min_group`` employee-years are dropped before the ratios are computed.
    """
    if not 0 < flag_fraction < 1:
        raise ValueError("flag_fraction must be between 0 and 1")
    table = fairness_audit(test, scores, column, flag_fraction)
    table = table[table["employee_years"] >= min_group]
    table.attrs["flag_rate_ratio"] = _ratio(table["flagged_pct"])
    table.attrs["false_flag_ratio"] = _ratio(table["false_flag_pct"])
    return table


def _ratio(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty or values.max() == 0:
        return float("nan")
    return float(values.min() / values.max())
