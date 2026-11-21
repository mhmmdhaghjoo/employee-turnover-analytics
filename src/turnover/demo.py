"""Synthetic HR records in the same format as the real dataset.

Used by the tests and CI, and by the app's demo mode when the real file is
not present. The employees, departments and effects are invented: short
tenure and Customer Service raise the resignation rate on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEARS = range(2006, 2016)


def make_raw(n_employees: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for employee_id in range(1, n_employees + 1):
        birth_year = int(rng.integers(1946, 1990))
        hire_year = int(max(birth_year + 19, rng.integers(1995, 2014)))
        department = rng.choice(["Customer Service", "Bakery", "Meats"], p=[0.3, 0.35, 0.35])
        gender = rng.choice(["Female", "Male"])
        exit_year, reason = None, None
        for year in YEARS:
            if year < hire_year:
                continue
            tenure, age_year_end = year - hire_year, year - birth_year
            if age_year_end in (60, 65) and rng.random() < 0.6:
                exit_year, reason = year, "Retirement"
            elif rng.random() < (0.12 if tenure < 2 else 0.01) * (
                2 if department == "Customer Service" else 1
            ):
                exit_year, reason = year, "Resignaton"
            elif year >= 2014 and rng.random() < 0.02:
                exit_year, reason = year, "Layoff"
            if exit_year:
                break
        last_year = exit_year or YEARS[-1]
        future_exit = f"6/15/{exit_year}" if exit_year else "1/1/1900"
        for year in range(max(hire_year, YEARS[0]), last_year + 1):
            leaving = year == exit_year
            rows.append(
                {
                    "EmployeeID": employee_id,
                    "recorddate_key": f"6/15/{year} 0:00" if leaving else f"12/31/{year} 0:00",
                    "birthdate_key": f"3/1/{birth_year}",
                    "orighiredate_key": f"2/1/{hire_year}",
                    "terminationdate_key": future_exit,  # present on ACTIVE rows too, as in the real file
                    "age": year - birth_year,
                    "length_of_service": year - hire_year,
                    "city_name": "Vancouver",
                    "department_name": department,
                    "job_title": "Cashier" if department == "Customer Service" else "Baker",
                    "store_name": int(employee_id % 6),
                    "gender_short": gender[0],
                    "gender_full": gender,
                    "termreason_desc": reason if leaving else "Not Applicable",
                    "termtype_desc": ("Involuntary" if reason == "Layoff" else "Voluntary")
                    if leaving
                    else "Not Applicable",
                    "STATUS_YEAR": year,
                    "STATUS": "TERMINATED" if leaving else "ACTIVE",
                    "BUSINESS_UNIT": "STORES",
                }
            )
    return pd.DataFrame(rows)
