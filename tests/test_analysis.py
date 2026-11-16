from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turnover.audit import fairness_audit
from turnover.data import build_panel, load_records
from turnover.describe import add_bands, rates_by, yearly_rates
from turnover.model import capture_at, split_by_year
from turnover.survival import life_table, median_tenure


@pytest.fixture(scope="module")
def panel(raw_path: Path) -> pd.DataFrame:
    return add_bands(build_panel(load_records(raw_path)))


def test_yearly_rates_add_up(panel: pd.DataFrame) -> None:
    rates = yearly_rates(panel)
    assert rates["headcount"].sum() == len(panel)
    assert rates["resignations"].sum() == panel["resigned"].sum()
    parts = rates[["resignation_rate_pct", "retirement_rate_pct", "layoff_rate_pct"]].sum(axis=1)
    assert np.allclose(parts, rates["total_exit_rate_pct"])


def test_wilson_interval_contains_the_rate_and_small_groups_are_dropped(panel: pd.DataFrame) -> None:
    table = rates_by(panel, "department", min_person_years=1)
    assert ((table["ci_low_pct"] <= table["rate_pct"]) & (table["rate_pct"] <= table["ci_high_pct"])).all()
    assert rates_by(panel, "department", min_person_years=10**9).empty


def test_life_table_by_hand() -> None:
    # 4 people in tenure year 0, one resigns; the 3 survivors reach year 1, one resigns.
    panel = pd.DataFrame({"tenure": [0.2, 0.5, 0.1, 0.9, 1.2, 1.5, 1.1], "resigned": [1, 0, 0, 0, 1, 0, 0]})
    table = life_table(panel, max_tenure=2)
    assert table.loc[0, "hazard"] == pytest.approx(0.25)
    assert table.loc[1, "hazard"] == pytest.approx(1 / 3)
    assert table.loc[1, "survival"] == pytest.approx(0.75 * 2 / 3)
    assert table.loc[2, "survival"] == pytest.approx(0.5)  # nobody at risk: survival carries over
    assert median_tenure(table) == 2.0


def test_survival_never_increases(panel: pd.DataFrame) -> None:
    survival = life_table(panel)["survival"]
    assert survival.is_monotonic_decreasing
    assert survival.between(0, 1).all()


def test_split_keeps_test_years_out_of_training(panel: pd.DataFrame) -> None:
    train, test = split_by_year(panel, 2014)
    assert train["year"].max() == 2013
    assert test["year"].min() == 2014
    assert panel["year"].min() not in set(train["year"])


def test_capture_of_a_perfect_ranking() -> None:
    y = np.array([1] * 5 + [0] * 95)
    assert capture_at(y, np.linspace(1, 0, 100), 0.05) == 1.0
    assert capture_at(y, np.linspace(0, 1, 100), 0.05) == 0.0


def test_fairness_audit_by_hand() -> None:
    test = pd.DataFrame({"gender": ["F"] * 5 + ["M"] * 5, "resigned": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]})
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1])
    table = fairness_audit(test, scores, "gender", flag_fraction=0.2)  # flags the two highest scores, both F
    assert table.loc["F", "flagged_pct"] == 40
    assert table.loc["F", "recall_pct"] == 100
    assert table.loc["F", "precision_pct"] == 50
    assert table.loc["M", "flagged_pct"] == 0
    assert table.loc["M", "recall_pct"] == 0
