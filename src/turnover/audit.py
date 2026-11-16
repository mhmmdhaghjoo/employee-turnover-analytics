"""Checks that decide whether the model's numbers can be trusted and used.

1. Target audit: how good does a model look if it predicts *any* exit, and
   how much of that is just the retirement age?
2. Fairness audit: does a risk list built from the model flag one gender or
   age group more often than its actual resignation rate justifies?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from turnover.model import FEATURES, _prepare, candidate_models


def target_audit(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """What happens if one model predicts *any* exit, as turnover models often do?

    The same model is scored separately on each exit reason (leavers of that
    reason against everyone who stayed). Three different processes hide
    behind the single label: retirement follows an age rule, layoffs are the
    employer's decision, and only resignation is the employee's own choice.
    """
    model = candidate_models(seed)["gradient_boosting"].fit(_prepare(train), train["terminated"].astype(int))
    scores = model.predict_proba(_prepare(test))[:, 1]
    exited = test["terminated"].to_numpy()
    stayed = ~exited

    rows = [{"scored on": "any exit", "leavers": int(exited.sum()), "roc_auc": roc_auc_score(exited, scores)}]
    for reason in ("retirement", "resignation", "layoff"):
        is_reason = test["exit_reason"].eq(reason).to_numpy()
        if not is_reason.any():
            continue
        subset = is_reason | stayed
        rows.append(
            {
                "scored on": f"{reason} vs. stayers",
                "leavers": int(is_reason.sum()),
                "roc_auc": roc_auc_score(is_reason[subset], scores[subset]),
            }
        )
    table = pd.DataFrame(rows).set_index("scored on")

    # The retirement rule: does the person turn 60 or 65 during the year?
    # Age on 31 December = age on 1 January + 1, rounded down.
    turns_60_or_65 = np.isin(np.floor(test["age"].to_numpy() + 1), [60, 65])
    retired = test["exit_reason"].eq("retirement").to_numpy()
    table.attrs["retirees_matching_age_rule"] = (
        float(turns_60_or_65[retired].mean()) if retired.any() else float("nan")
    )
    table.attrs["age_rule_precision"] = (
        float(retired[turns_60_or_65].mean()) if turns_60_or_65.any() else float("nan")
    )
    table.attrs["features"] = FEATURES
    return table


def fairness_audit(
    test: pd.DataFrame, scores: np.ndarray, column: str, flag_fraction: float = 0.10
) -> pd.DataFrame:
    """Per group: how many are flagged in the top ``flag_fraction``, and how often the flag is right.

    A group is treated unfairly by a risk list if it is flagged much more
    often than others *without* resigning more often (low precision), or if
    its leavers are missed more often (low recall).
    """
    threshold = np.quantile(scores, 1 - flag_fraction)
    frame = pd.DataFrame(
        {
            "group": test[column].to_numpy(),
            "resigned": test["resigned"].to_numpy(),
            "flagged": scores >= threshold,
        }
    )
    rows = []
    for name, group in frame.groupby("group"):
        flagged, resigned = group["flagged"], group["resigned"].astype(bool)
        rows.append(
            {
                column: name,
                "employee_years": len(group),
                "resignation_rate_pct": 100 * resigned.mean(),
                "flagged_pct": 100 * flagged.mean(),
                "precision_pct": 100 * resigned[flagged].mean() if flagged.any() else np.nan,
                "recall_pct": 100 * flagged[resigned].mean() if resigned.any() else np.nan,
                "false_flag_pct": 100 * flagged[~resigned].mean(),
            }
        )
    table = pd.DataFrame(rows).set_index(column)
    table.attrs["flag_rate_ratio"] = float(table["flagged_pct"].min() / table["flagged_pct"].max())
    table.attrs["false_flag_ratio"] = float(table["false_flag_pct"].min() / table["false_flag_pct"].max())
    return table
