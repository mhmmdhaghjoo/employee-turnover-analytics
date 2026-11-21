"""People analytics explorer: when and where people resign, and whether a risk list is fair.

Run from the repository root:

    streamlit run app/streamlit_app.py

With data/MFG10YearTerminationData.csv present it uses the public practice
dataset; otherwise it runs on synthetic records and says so on every page.
It shows group-level results only. It does not score or list individuals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from turnover.explore import group_fairness, load_panel, stratified_rates, survival_curves
from turnover.model import fit_and_score, split_by_year

DATA = Path("data/MFG10YearTerminationData.csv")
ACCENT = "#2b6cb0"
ATTRIBUTES = {
    "Department": "department",
    "Tenure band": "tenure_band",
    "Age band": "age_band",
    "Gender": "gender",
    "Business unit": "business_unit",
}
MODELS = {
    "Tenure rule (baseline)": "tenure_only",
    "Logistic regression": "logistic_regression",
    "Gradient boosting": "gradient_boosting",
}

st.set_page_config(page_title="People analytics explorer", layout="wide")


@st.cache_data(show_spinner="Loading records ...")
def get_panel() -> tuple[pd.DataFrame, bool]:
    return load_panel(DATA)


@st.cache_resource(show_spinner="Training on earlier years, scoring unseen years ...")
def get_scores(first_test_year: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    panel, _ = get_panel()
    train, test = split_by_year(panel, first_test_year)
    _, scores, comparison = fit_and_score(train, test)
    return test, scores, comparison


panel, is_demo = get_panel()

st.title("People analytics explorer")
if is_demo:
    st.warning(
        "Demo mode: synthetic employees generated in `turnover/demo.py`. Put the public practice "
        f"dataset at `{DATA}` to see the real analysis. Nothing here describes a real company."
    )
else:
    st.caption(
        "Public practice HR dataset (MFG10YearTerminationData, widely shared on Kaggle), "
        f"{panel['employee_id'].nunique():,} employees, {panel['year'].min()} to {panel['year'].max()}. "
        "Its origin is not documented, so treat it as illustrative, not as a real employer."
    )
st.info(
    "Group-level results only. The app does not show individual risk scores: a resignation "
    "forecast should inform where to look (onboarding, a team, a role), never a decision about a person.",
    icon="ℹ️",
)

tab_survival, tab_strata, tab_fairness, tab_check = st.tabs(
    ["Cohort survival", "Where resignations concentrate", "Fairness of a risk list", "How to check this"]
)

with tab_survival:
    left, right = st.columns([1, 3])
    with left:
        label = st.selectbox("Compare by", ["None", *ATTRIBUTES], index=1)
        max_tenure = st.slider("Years of tenure", 3, 15, 10)
        min_at_risk = st.slider("Hide years with fewer employees at risk than", 10, 200, 50, 10)
    column = None if label == "None" else ATTRIBUTES[label]
    curves = survival_curves(panel, column, max_tenure, min_at_risk)
    with right:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for name, group in curves.groupby("group"):
            x = group["tenure_year"] + 1
            (line,) = ax.step(x, group["survival"], where="post", label=name)
            ax.fill_between(x, group["low"], group["high"], step="post", alpha=0.12, color=line.get_color())
        ax.set_xlabel("Completed years of tenure")
        ax.set_ylabel("Share not yet resigned")
        ax.set_ylim(max(0.0, curves["low"].min() - 0.05) if len(curves) else 0, 1.0)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Life-table estimate with 95 % pointwise intervals (Greenwood). Retirements and layoffs "
            "end the observation but are not counted as resignations. Curves stop where too few "
            "employees remain at risk."
        )

with tab_strata:
    c1, c2, c3 = st.columns(3)
    rows_label = c1.selectbox("Rows", list(ATTRIBUTES), index=0)
    cols_label = c2.selectbox("Columns", list(ATTRIBUTES), index=1)
    min_years = c3.slider("Suppress cells with fewer employee-years than", 20, 500, 100, 20)
    if rows_label == cols_label:
        st.error("Pick two different attributes.")
    else:
        table = stratified_rates(panel, ATTRIBUTES[rows_label], ATTRIBUTES[cols_label], min_years)
        pivot = table.pivot(index=ATTRIBUTES[rows_label], columns=ATTRIBUTES[cols_label], values="rate_pct")
        counts = table.pivot(
            index=ATTRIBUTES[rows_label], columns=ATTRIBUTES[cols_label], values="person_years"
        )
        st.subheader("Resignations per 100 employee-years")
        st.dataframe(
            pivot.style.format("{:.1f}", na_rep="suppressed").background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )
        with st.expander("Employee-years behind each cell"):
            st.dataframe(counts.fillna(0).astype(int), use_container_width=True)
        st.caption(
            "Rates are not adjusted for each other: a department with many new hires looks worse "
            "partly because of tenure. Compare within a tenure band before drawing conclusions."
        )

with tab_fairness:
    years = sorted(panel["year"].unique())
    c1, c2, c3, c4 = st.columns(4)
    first_test_year = c1.selectbox("Test on years from", years[2:], index=len(years[2:]) - 2)
    model_label = c2.selectbox("Risk model", list(MODELS), index=2)
    attribute_label = c3.selectbox("Check groups by", ["Gender", "Age band"], index=0)
    flag_pct = c4.slider("Flag the top ... % of employee-years", 1, 30, 10)
    test, scores, comparison = get_scores(int(first_test_year))
    if test["resigned"].sum() == 0:
        st.error("No resignations in the test years; choose an earlier start year.")
    else:
        table = group_fairness(test, scores[MODELS[model_label]], ATTRIBUTES[attribute_label], flag_pct / 100)
        m1, m2, m3 = st.columns(3)
        m1.metric("ROC AUC on unseen years", f"{comparison.loc[MODELS[model_label], 'roc_auc']:.3f}")
        m2.metric("Flag-rate ratio (1 = parity)", f"{table.attrs['flag_rate_ratio']:.2f}")
        m3.metric("False-flag ratio (1 = parity)", f"{table.attrs['false_flag_ratio']:.2f}")
        shown = table.rename(
            columns={
                "employee_years": "Employee-years",
                "resignation_rate_pct": "Resigned %",
                "flagged_pct": "Flagged %",
                "precision_pct": "Flag correct %",
                "recall_pct": "Leavers found %",
                "false_flag_pct": "Stayers flagged %",
            }
        )
        st.dataframe(
            shown.style.format("{:.1f}", na_rep="-").format("{:,.0f}", subset=["Employee-years"]),
            use_container_width=True,
        )
        n_leavers = int(test["resigned"].sum())
        st.caption(
            f"Test years contain {len(test):,} employee-years and only {n_leavers} resignations, so "
            "'Leavers found %' rests on a few people per group and moves a lot between settings. "
            "Gender and age are not model features; they are used only for this check."
        )
        st.markdown(
            "**Reading it.** A fair-looking flag rate can hide unequal errors: compare *Leavers found* "
            "between groups. In the full report the tenure rule finds every man who resigns but 44 % "
            "of the women; the leavers it misses are all women with five to nine years of service."
            if not is_demo
            else "**Reading it.** Compare *Leavers found* between groups, not only *Flagged*: "
            "a list can flag groups equally often and still miss one group's leavers."
        )

with tab_check:
    st.markdown(
        """
**What I check before trusting a people analytics result, mine or an AI assistant's**

1. **Who left, and why.** Resignations, retirements and layoffs are different outcomes. Mixing them
   makes age look like a resignation driver when it is a retirement rule.
2. **Leakage.** The raw file has a termination date on *active* rows too; a model that sees it
   "predicts" perfectly. The repository's target audit tests for this.
3. **Time.** Train on earlier years, test on later ones, and resample employees (not rows) for
   intervals, because one person appears in many years.
4. **Small groups.** Suppress cells with little data; a 0 % or 50 % rate from twenty people means little.
5. **Fairness.** Compare errors per group, not only accuracy overall, and decide in advance what
   gap is acceptable.
6. **Use.** A forecast points to where onboarding or management needs attention. It should not be
   used to rate, promote or dismiss individuals. In Germany that would also raise GDPR questions
   (Art. 22 on automated decisions) and works-council co-determination (BetrVG § 87); I would
   involve the data protection officer and the works council before any such use.
"""
    )
