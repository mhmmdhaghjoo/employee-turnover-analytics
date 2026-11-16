from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turnover.data import DataError, build_panel, load_records


def test_one_row_per_employee_year_and_one_exit_per_employee(raw_path: Path) -> None:
    records = load_records(raw_path)
    assert not records.duplicated(["employee_id", "year"]).any()
    assert records.loc[records["terminated"]].groupby("employee_id").size().max() == 1
    assert set(records["exit_reason"]) == {"none", "resignation", "retirement", "layoff"}


def test_future_termination_date_is_not_carried_into_the_panel(raw_path: Path) -> None:
    panel = build_panel(load_records(raw_path))
    assert not any("termination" in column for column in panel.columns)


def test_age_and_tenure_are_measured_at_the_start_of_the_year() -> None:
    records = pd.DataFrame(
        {
            "employee_id": [1, 1],
            "year": [2010, 2011],
            "birth_date": pd.to_datetime(["1980-01-01"] * 2),
            "hire_date": pd.to_datetime(["2010-07-01"] * 2),
            "city": "X", "department": "D", "job_title": "Cashier", "store": "1", "gender": "Female",
            "business_unit": "STORES", "exit_reason": ["none", "resignation"], "terminated": [False, True],
        }
    )  # fmt: skip
    panel = build_panel(records).set_index("year")
    assert panel.loc[2010, "tenure"] == 0  # hired later that year: clipped at zero
    assert panel.loc[2010, "new_hire"]
    assert panel.loc[2011, "tenure"] == pytest.approx(184 / 365.25)
    assert panel.loc[2011, "age"] == pytest.approx(31, abs=0.01)
    assert panel["resigned"].tolist() == [0, 1]


def test_team_rates_come_from_the_previous_year_only(raw_path: Path) -> None:
    panel = build_panel(load_records(raw_path))
    first_year = panel["year"].min()
    assert panel.loc[panel["year"] == first_year, "store_resign_rate_prev"].isna().all()
    # Recompute one cell by hand.
    store, year = panel.loc[panel["year"] == 2010, ["store", "year"]].iloc[0]
    expected = panel.loc[(panel["store"] == store) & (panel["year"] == year - 1), "resigned"].mean()
    actual = panel.loc[(panel["store"] == store) & (panel["year"] == year), "store_resign_rate_prev"].iloc[0]
    assert actual == pytest.approx(expected)


def test_missing_file_and_missing_columns_raise_clear_errors(tmp_path: Path, raw_path: Path) -> None:
    with pytest.raises(DataError, match="README"):
        load_records(tmp_path / "nope.csv")
    broken = tmp_path / "broken.csv"
    pd.read_csv(raw_path).drop(columns="STATUS").to_csv(broken, index=False)
    with pytest.raises(DataError, match="STATUS"):
        load_records(broken)
