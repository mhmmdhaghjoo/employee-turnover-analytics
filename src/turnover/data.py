"""Load the yearly HR snapshot file and turn it into an analysis-ready panel.

The raw file has one row per employee and year: an ACTIVE row dated
31 December for everyone still employed, and a TERMINATED row in the year
somebody leaves. Two properties of the file need care:

* ACTIVE rows already carry the employee's *future* termination date. Using
  that column as a feature would leak the outcome, so it is dropped here.
* ``age`` and ``length_of_service`` are measured on the record date, which is
  the termination date for leavers and year-end for stayers. To compare like
  with like, both are recomputed as of 1 January of each year.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "EmployeeID", "recorddate_key", "birthdate_key", "orighiredate_key", "city_name",
    "department_name", "job_title", "store_name", "gender_full", "termreason_desc",
    "termtype_desc", "STATUS_YEAR", "STATUS", "BUSINESS_UNIT",
}  # fmt: skip

# The source file spells "Resignaton"; map every reason to a clean label.
EXIT_REASONS = {"Resignaton": "resignation", "Retirement": "retirement", "Layoff": "layoff"}


class DataError(ValueError):
    """The input file does not have the structure the analysis relies on."""


def load_records(path: Path) -> pd.DataFrame:
    """Read the raw file, validate it and return one clean row per employee-year."""
    if not path.exists():
        raise DataError(f"{path} not found. See the README for where to get the dataset.")
    raw = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise DataError(f"missing columns: {sorted(missing)}")

    records = pd.DataFrame(
        {
            "employee_id": raw["EmployeeID"],
            "year": raw["STATUS_YEAR"].astype(int),
            "birth_date": pd.to_datetime(raw["birthdate_key"]),
            "hire_date": pd.to_datetime(raw["orighiredate_key"]),
            "city": raw["city_name"],
            "department": raw["department_name"],
            "job_title": raw["job_title"],
            "store": raw["store_name"].astype(str),
            "gender": raw["gender_full"],
            "business_unit": raw["BUSINESS_UNIT"],
            "exit_reason": raw["termreason_desc"].map(EXIT_REASONS).fillna("none"),
            "terminated": raw["STATUS"].eq("TERMINATED"),
        }
    )
    unknown = set(raw.loc[records["terminated"], "termreason_desc"]) - set(EXIT_REASONS)
    if unknown:
        raise DataError(f"unknown termination reasons: {sorted(unknown)}")
    if (records["hire_date"] < records["birth_date"]).any():
        raise DataError("hire date before birth date")

    # A few leavers also have an ACTIVE row in their exit year; the exit wins.
    records = records.sort_values("terminated").drop_duplicates(["employee_id", "year"], keep="last")
    if records.loc[records["terminated"]].duplicated("employee_id").any():
        raise DataError("an employee is terminated more than once")

    logger.info(
        "loaded %d employee-years, %d employees, %d exits",
        len(records),
        records["employee_id"].nunique(),
        int(records["terminated"].sum()),
    )
    return records.sort_values(["employee_id", "year"], ignore_index=True)


def build_panel(records: pd.DataFrame) -> pd.DataFrame:
    """Add the features known on 1 January of each year, and the outcome of that year."""
    panel = records.copy()
    year_start = pd.to_datetime(panel["year"].astype(str) + "-01-01")
    panel["age"] = (year_start - panel["birth_date"]).dt.days / 365.25
    panel["tenure"] = ((year_start - panel["hire_date"]).dt.days / 365.25).clip(lower=0)
    panel["log_tenure"] = np.log1p(panel["tenure"])
    panel["new_hire"] = panel["hire_date"].dt.year.eq(panel["year"])
    panel["age_at_hire"] = (panel["hire_date"] - panel["birth_date"]).dt.days / 365.25
    panel["is_manager"] = panel["job_title"].str.contains("Manager|Director|VP|CEO|Chief|Exec", regex=True)
    panel["resigned"] = panel["exit_reason"].eq("resignation").astype(int)

    # Team context from the previous year only: last year's resignation rate
    # in the same store and in the same department.
    for level in ("store", "department"):
        rate = panel.groupby([level, "year"])["resigned"].mean().rename(f"{level}_resign_rate_prev")
        rate = rate.reset_index()
        rate["year"] += 1
        panel = panel.merge(rate, on=[level, "year"], how="left")
    panel["store_headcount"] = panel.groupby(["store", "year"])["employee_id"].transform("size")
    return panel.drop(columns=["birth_date", "hire_date"])
