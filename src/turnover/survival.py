"""How long do people stay before they resign?

A discrete-time life table: for each completed year of tenure, the hazard is
the share of employees at risk in that tenure year who resign during it.
Because it is built from employee-years, it handles the two complications of
HR data without extra machinery: people who were already employed when the
data starts (they only enter the risk set at their current tenure) and people
who leave for other reasons or are still employed (they simply stop
contributing years).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def life_table(panel: pd.DataFrame, max_tenure: int = 15) -> pd.DataFrame:
    """Resignation hazard and survival by completed years of tenure."""
    years = panel.assign(tenure_year=np.floor(panel["tenure"]).astype(int))
    years = years[years["tenure_year"] <= max_tenure]
    table = years.groupby("tenure_year")["resigned"].agg(at_risk="size", resignations="sum")
    table = table.reindex(range(max_tenure + 1), fill_value=0)
    table["hazard"] = (table["resignations"] / table["at_risk"].where(table["at_risk"] > 0)).fillna(0.0)
    # Survival to the END of each tenure year.
    table["survival"] = (1 - table["hazard"]).cumprod()
    # Greenwood's formula for the standard error of the survival estimate.
    safe = table["at_risk"] * (table["at_risk"] - table["resignations"])
    variance_terms = (table["resignations"] / safe.where(safe > 0)).fillna(0.0)
    table["survival_se"] = table["survival"] * np.sqrt(variance_terms.cumsum())
    return table


def life_tables_by(panel: pd.DataFrame, column: str, max_tenure: int = 15) -> pd.DataFrame:
    """One life table per group, stacked."""
    tables = {name: life_table(group, max_tenure) for name, group in panel.groupby(column)}
    return pd.concat(tables, names=[column])


def median_tenure(table: pd.DataFrame) -> float | None:
    """First tenure year by whose end half of a cohort has resigned; None if never reached."""
    below = table.index[table["survival"] <= 0.5]
    return float(below[0] + 1) if len(below) else None
