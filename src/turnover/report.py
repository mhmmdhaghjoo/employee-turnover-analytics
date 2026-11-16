"""Write the HTML management report. Every number in the text comes from the run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment

_ENV = Environment(autoescape=False)
_ENV.filters["fmt"] = lambda value: f"{value:,}"

TEMPLATE = _ENV.from_string(
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Employee turnover report</title>
<style>
 body{font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:860px;margin:0 auto;padding:32px 20px 64px}
 h1{font-size:30px;line-height:1.2;margin:0 0 4px} h2{font-size:21px;margin:40px 0 8px;padding-top:16px;border-top:1px solid #e1e0d9}
 .sub{color:#6b6a65;margin:0 0 24px} img{max-width:100%;height:auto;border:1px solid #e1e0d9;border-radius:6px;margin:12px 0}
 .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}
 .kpi{border:1px solid #e1e0d9;border-radius:8px;padding:14px 16px;background:#fff}
 .kpi b{display:block;font-size:26px;line-height:1.2} .kpi span{color:#6b6a65;font-size:14px}
 table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14.5px;font-variant-numeric:tabular-nums}
 th,td{padding:7px 10px;border-bottom:1px solid #e1e0d9;text-align:right} th:first-child,td:first-child{text-align:left}
 th{color:#6b6a65;font-weight:600} .note{color:#6b6a65;font-size:14px} li{margin:6px 0}
</style></head><body>
<h1>Who leaves, why, and can we see it coming?</h1>
<p class="sub">Turnover analysis of {{ s.employees | fmt }} employees, {{ s.years[0] }} to {{ s.years[1] }}
 ({{ s.employee_years | fmt }} employee-years)</p>

<div class="kpis">
 <div class="kpi"><b>{{ "%.1f" | format(s.mean_total_exit_rate_pct) }}%</b><span>of staff leave in an average year, for any reason</span></div>
 <div class="kpi"><b>{{ "%.1f" | format(s.mean_resignation_rate_pct) }}%</b><span>resign in an average year</span></div>
 <div class="kpi"><b>{{ "%.1f" | format(s.first_year_rate_pct) }}%</b><span>resign in their first year, against {{ "%.1f" | format(s.ten_plus_rate_pct) }}% after ten years</span></div>
 <div class="kpi"><b>{{ "%.1f" | format(s.top_department_rate_pct) }}%</b><span>resign per year in {{ s.top_department }}, against {{ "%.1f" | format(s.other_departments_rate_pct) }}% elsewhere</span></div>
</div>

<h2>1. Three different things are called "turnover"</h2>
<p>Of {{ (s.exits.resignation + s.exits.retirement + s.exits.layoff) | fmt }} exits, {{ s.exits.retirement | fmt }} were
retirements, {{ s.exits.resignation | fmt }} resignations and {{ s.exits.layoff | fmt }} layoffs. Only resignations are the
employee's own decision and can be influenced by retention measures, so the rest of this report is about them.</p>
<img src="figures/exits_by_year.png" alt="Exits per year by reason">
<p>Retirement needs no model: {{ "%.0f" | format(s.retirees_matching_age_rule_pct) }}% of retirees in the test years left
in the year they turned 60 or 65. Layoffs are a company decision and cannot be predicted from employee data.</p>

<h2>2. Resignations happen early, and mostly in one department</h2>
<img src="figures/by_tenure.png" alt="Resignation rate by tenure">
<img src="figures/by_department.png" alt="Resignation rate by department">
<p>After five years, {{ "%.0f" | format(s.survival_5y_pct) }}% of a hiring cohort has not resigned; in
{{ s.top_department }} it is {{ "%.0f" | format(s.survival_5y_top_department_pct) }}%.</p>
<img src="figures/survival.png" alt="Survival curves">

<h2>3. Can next year's resignations be predicted?</h2>
<p>Models were fitted on {{ s.train_years[0] }} to {{ s.train_years[1] }} and tested on {{ s.test_years[0] }} to
{{ s.test_years[1] }}, years they had never seen ({{ s.test_employee_years | fmt }} employee-years,
{{ s.test_resignations }} resignations).</p>
{{ comparison }}
<img src="figures/capture.png" alt="Share of resignations found by share of employees reviewed">
<p class="note">ROC AUC: 0.5 = guessing, 1.0 = perfect. "Found in top 10 %" = share of all resignations among the 10 % of
employees with the highest predicted risk. Intervals resample employees, not rows.</p>

<h2>4. How far can these numbers be trusted and used?</h2>
<ul>
 <li><b>A flag is usually wrong.</b> Resignation is rare, so among the 10% of employees the best model
 ({{ s.best_model | replace("_", " ") }}) ranks highest, only {{ "%.1f" | format(s.precision_top_10pct_pct) }}% actually resign.
 The list shows where to look, not who will leave.</li>
 <li><b>One label hides three processes.</b> A model trained to predict any exit scores
 {{ "%.2f" | format(s.target_audit_auc["any exit"]) }} overall, which mixes near-certainty for retirement
 ({{ "%.2f" | format(s.target_audit_auc["retirement vs. stayers"]) }}){% if "layoff vs. stayers" in s.target_audit_auc %}
 with guessing for layoffs ({{ "%.2f" | format(s.target_audit_auc["layoff vs. stayers"]) }}){% endif %}.</li>
 <li><b>The raw file leaks the answer.</b> Records of active employees already contain their future termination date. That
 column is dropped before any modelling.</li>
 <li><b>Gender is not a model input, but outcomes still differ by gender.</b> See the table: the share of leavers a
 10% risk list finds is not the same for women and men.</li>
</ul>
<p><b>Best model ({{ s.best_model | replace("_", " ") }}), top 10% flagged</b></p>
{{ fairness_best }}
<p><b>Simple rule (shortest tenure first), top 10% flagged</b></p>
{{ fairness_rule }}

<h2>5. What I would do with this</h2>
<ul>
 <li>Focus retention work on the first two years and on {{ s.top_department }}: onboarding, scheduling and pay there
 affect the most leavers.</li>
 <li>Use predicted risk for team-level planning (expected leavers per store and department), not for decisions about
 individuals. Age and tenure drive the scores, and age is a protected characteristic.</li>
 <li>Plan retirements from the age structure directly; no model is needed.</li>
 <li>Before any operational use: agree with the works council and data protection officer what is scored and who sees
 it, keep a human decision in every case, and re-check the gender table each year.</li>
</ul>
<p class="note">Generated by the turnover package. Method details are in the repository README.</p>
</body></html>
"""
)


def _html_table(table: pd.DataFrame, columns: dict[str, str], digits: int = 1) -> str:
    shown = table[list(columns)].rename(columns=columns)
    shown.index = [str(i).replace("_", " ") for i in shown.index]
    return shown.to_html(float_format=lambda v: f"{v:.{digits}f}", border=0, na_rep="n/a")


def render_report(summary: dict[str, object], tables: dict[str, pd.DataFrame], path: Path) -> None:
    comparison = tables["model_comparison"].copy()
    for column in ("capture_top_5pct", "capture_top_10pct", "capture_top_20pct"):
        comparison[column] *= 100
    comparison_html = _html_table(
        comparison,
        {
            "roc_auc": "ROC AUC",
            "roc_auc_ci_low": "95% low",
            "roc_auc_ci_high": "95% high",
            "pr_auc": "PR AUC",
            "capture_top_5pct": "Found in top 5% (%)",
            "capture_top_10pct": "Found in top 10% (%)",
            "capture_top_20pct": "Found in top 20% (%)",
        },  # fmt: skip
        digits=3,
    )
    fairness_columns = {
        "employee_years": "Employee-years", "resignation_rate_pct": "Resign (%)", "flagged_pct": "Flagged (%)",
        "recall_pct": "Leavers found (%)", "precision_pct": "Flags that were right (%)",
        "false_flag_pct": "Stayers flagged (%)",
    }  # fmt: skip
    html = TEMPLATE.render(
        s=summary,
        comparison=comparison_html,
        fairness_best=_html_table(tables["fairness_gender"], fairness_columns),
        fairness_rule=_html_table(tables["fairness_gender_tenure_rule"], fairness_columns),
    )
    path.write_text(html, encoding="utf-8")
