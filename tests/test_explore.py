from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turnover.data import build_panel, load_records
from turnover.describe import add_bands
from turnover.explore import group_fairness, load_panel, stratified_rates, survival_curves
from turnover.model import split_by_year


@pytest.fixture(scope="module")
def panel(raw_path: Path) -> pd.DataFrame:
    return add_bands(build_panel(load_records(raw_path)))


def _tiny_panel() -> pd.DataFrame:
    # 4 people start; one resigns in year 0, one in year 1; the others stay.
    return pd.DataFrame(
        {
            "tenure": [0.2, 0.3, 0.4, 0.5, 1.2, 1.3, 1.4, 2.1, 2.2],
            "resigned": [1, 0, 0, 0, 1, 0, 0, 0, 0],
            "dept": ["a"] * 9,
        }
    )


def test_survival_curve_by_hand() -> None:
    curve = survival_curves(_tiny_panel(), min_at_risk=1).set_index("tenure_year")
    assert curve.loc[0, "survival"] == pytest.approx(3 / 4)
    assert curve.loc[1, "survival"] == pytest.approx(3 / 4 * 2 / 3)
    assert curve.loc[2, "survival"] == pytest.approx(0.5)
    assert ((curve["low"] <= curve["survival"]) & (curve["survival"] <= curve["high"])).all()
    assert curve["low"].min() >= 0 and curve["high"].max() <= 1


def test_survival_curve_stops_where_data_is_thin() -> None:
    curve = survival_curves(_tiny_panel(), min_at_risk=3)
    assert curve["tenure_year"].tolist() == [0, 1]  # year 2 has only 2 people at risk


def test_survival_curves_per_group(panel: pd.DataFrame) -> None:
    curves = survival_curves(panel, "department")
    assert set(curves["group"]) <= set(panel["department"])
    first_year = curves[curves["tenure_year"] == 0].set_index("group")["survival"]
    # the synthetic data doubles early resignations in Customer Service
    assert first_year["Customer Service"] < first_year.drop("Customer Service").min()


def test_stratified_rates_suppresses_small_cells(panel: pd.DataFrame) -> None:
    table = stratified_rates(panel, "department", "tenure_band", min_person_years=100)
    assert table["person_years"].sum() == len(panel)
    small = table[table["person_years"] < 100]
    assert small["rate_pct"].isna().all() and small["suppressed"].all()
    big = table[~table["suppressed"]].iloc[0]
    assert big["rate_pct"] == pytest.approx(100 * big["resignations"] / big["person_years"])


def test_group_fairness_ratios(panel: pd.DataFrame) -> None:
    _, test = split_by_year(panel, 2014)
    scores = np.random.default_rng(0).uniform(size=len(test))  # risk unrelated to anything
    table = group_fairness(test, scores, "gender", flag_fraction=0.2)
    assert set(table.index) == {"Female", "Male"}
    # random scores flag both groups at about the same rate
    assert table.attrs["flag_rate_ratio"] > 0.7
    with pytest.raises(ValueError):
        group_fairness(test, scores, "gender", flag_fraction=1.5)


def test_load_panel_falls_back_to_demo(tmp_path: Path) -> None:
    panel, is_demo = load_panel(tmp_path / "missing.csv")
    assert is_demo
    assert {"tenure_band", "age_band", "resigned"} <= set(panel.columns)
