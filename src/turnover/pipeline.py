"""Run the whole analysis and write tables, figures and the HTML report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from turnover import figures
from turnover.audit import fairness_audit, target_audit
from turnover.data import build_panel, load_records
from turnover.describe import TENURE_LABELS, add_bands, rates_by, yearly_rates
from turnover.model import coefficient_table, fit_and_score, split_by_year
from turnover.report import render_report
from turnover.survival import life_table

logger = logging.getLogger(__name__)


def run(data_path: Path, out_dir: Path, first_test_year: int = 2014, seed: int = 42) -> dict[str, object]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = add_bands(build_panel(load_records(data_path)))

    # 1. Where and why do people leave?
    rates = yearly_rates(panel)
    by_department = rates_by(panel, "department")
    by_tenure = rates_by(panel, "tenure_band")
    by_age = rates_by(panel, "age_band")
    by_gender = rates_by(panel, "gender")

    # 2. How long do they stay?
    top_department = by_department.index[0]
    survival_all = life_table(panel)
    survival_top = life_table(panel[panel["department"] == top_department])
    survival_rest = life_table(panel[panel["department"] != top_department])

    # 3. Can next year's resignations be predicted? Tested on unseen years.
    train, test = split_by_year(panel, first_test_year)
    fitted, scores, comparison = fit_and_score(train, test, seed)
    full_models = comparison.drop(index="tenure_only")
    best = str(full_models["roc_auc"].idxmax())
    coefficients = coefficient_table(fitted["logistic_regression"])

    # 4. Can the numbers be trusted and used?
    targets = target_audit(train, test, seed)
    fairness = {
        column: fairness_audit(test, scores[best], column, flag_fraction=0.10)
        for column in ("gender", "age_band")
    }
    fairness_baseline = fairness_audit(test, scores["tenure_only"], "gender", flag_fraction=0.10)

    tables = {
        "yearly_rates": rates, "by_department": by_department, "by_tenure": by_tenure, "by_age": by_age,
        "by_gender": by_gender, "survival": survival_all, "model_comparison": comparison,
        "logistic_coefficients": coefficients.set_index("feature"), "target_audit": targets,
        "fairness_gender": fairness["gender"], "fairness_age": fairness["age_band"],
        "fairness_gender_tenure_rule": fairness_baseline,
    }  # fmt: skip
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv")

    figures.exits_by_year(rates, fig_dir / "exits_by_year.png")
    figures.rate_bars(by_tenure, "Resignation rate by tenure at the start of the year", fig_dir / "by_tenure.png",
                      order=TENURE_LABELS[::-1])  # fmt: skip
    figures.rate_bars(by_department, "Resignation rate by department", fig_dir / "by_department.png")
    figures.survival_curves(
        {top_department: survival_top, "All other departments": survival_rest}, fig_dir / "survival.png"
    )
    figures.capture_curves(test["resigned"].to_numpy(), scores, fig_dir / "capture.png")

    test_resignations = int(test["resigned"].sum())
    flagged = fairness["gender"]["employee_years"] * fairness["gender"]["flagged_pct"] / 100
    summary: dict[str, object] = {
        "employees": int(panel["employee_id"].nunique()),
        "employee_years": int(len(panel)),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "exits": {
            reason: int((panel["exit_reason"] == reason).sum())
            for reason in ("resignation", "retirement", "layoff")
        },
        "mean_total_exit_rate_pct": float(rates["total_exit_rate_pct"].mean()),
        "mean_resignation_rate_pct": float(rates["resignation_rate_pct"].mean()),
        "top_department": top_department,
        "top_department_rate_pct": float(by_department["rate_pct"].iloc[0]),
        "other_departments_rate_pct": float(
            100 * panel.loc[panel["department"] != top_department, "resigned"].mean()
        ),
        "first_year_rate_pct": float(by_tenure.loc["under 1 yr", "rate_pct"]),
        "ten_plus_rate_pct": float(by_tenure.loc["10+ yrs", "rate_pct"]),
        "survival_5y_pct": float(100 * survival_all.loc[4, "survival"]),
        "survival_5y_top_department_pct": float(100 * survival_top.loc[4, "survival"]),
        "train_years": [int(train["year"].min()), int(train["year"].max())],
        "test_years": [int(test["year"].min()), int(test["year"].max())],
        "test_employee_years": int(len(test)),
        "test_resignations": test_resignations,
        "best_model": best,
        "metrics": json.loads(comparison.to_json(orient="index")),
        "precision_top_10pct_pct": float(
            100 * (fairness["gender"]["precision_pct"] / 100 * flagged).sum() / flagged.sum()
        ),
        "retirees_matching_age_rule_pct": float(100 * targets.attrs["retirees_matching_age_rule"]),
        "target_audit_auc": {str(k): float(v) for k, v in targets["roc_auc"].items()},
        "fairness_gender": json.loads(fairness["gender"].to_json(orient="index")),
        "fairness_gender_tenure_rule": json.loads(fairness_baseline.to_json(orient="index")),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_report(summary, tables, out_dir / "report.html")
    logger.info("best model %s; report written to %s", best, out_dir / "report.html")
    return summary
