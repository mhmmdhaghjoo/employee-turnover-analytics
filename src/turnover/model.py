"""Predict who resigns in the coming year, and test it on years the model never saw."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "age", "tenure", "log_tenure", "age_at_hire", "store_headcount",
    "store_resign_rate_prev", "department_resign_rate_prev",
]  # fmt: skip
BOOLEAN = ["new_hire", "is_manager"]
CATEGORICAL = ["department", "business_unit"]
FEATURES = NUMERIC + BOOLEAN + CATEGORICAL
# Gender is deliberately not a feature. It is only used afterwards, to check
# whether the model treats women and men differently (see audit.py).
TARGET = "resigned"


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    X = panel[FEATURES].copy()
    X[BOOLEAN] = X[BOOLEAN].astype(float)
    return X


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC + BOOLEAN),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        # Baseline a manager could apply without a model: short tenure = higher risk.
        "tenure_only": Pipeline(
            [
                ("select", ColumnTransformer([("tenure", "passthrough", ["tenure"])])),
                ("log", _LogTransform()),
                ("model", LogisticRegression()),
            ]
        ),
        "logistic_regression": Pipeline(
            [("prep", _preprocessor(scale=True)), ("model", LogisticRegression(max_iter=2000, C=0.5))]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed
                    ),
                ),
            ]
        ),
    }


class _LogTransform:
    """log(1 + x); a tiny transformer so the baseline stays a single pipeline."""

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        return self

    def transform(self, X):  # noqa: ANN001, ANN201
        return np.log1p(np.asarray(X, dtype=float))


def split_by_year(panel: pd.DataFrame, first_test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier years, test on later ones. The first data year has no
    previous-year team rates and is used only to compute them."""
    usable = panel[panel["year"] > panel["year"].min()]
    return usable[usable["year"] < first_test_year], usable[usable["year"] >= first_test_year]


def capture_at(y: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Share of all resignations found in the top ``fraction`` of scores."""
    n_top = max(1, int(round(len(y) * fraction)))
    top = np.argsort(-scores, kind="stable")[:n_top]
    return float(y[top].sum() / y.sum())


def evaluate(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "capture_top_5pct": capture_at(y, scores, 0.05),
        "capture_top_10pct": capture_at(y, scores, 0.10),
        "capture_top_20pct": capture_at(y, scores, 0.20),
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str = "roc_auc",
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """95 % interval, resampling employees (not rows) because one person appears in several years."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows_of = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.asarray)
    function = roc_auc_score if metric == "roc_auc" else average_precision_score
    values = []
    while len(values) < n_boot:
        idx = np.concatenate(rows_of.loc[rng.choice(unique, len(unique))].to_numpy())
        if y[idx].min() == y[idx].max():
            continue
        values.append(function(y[idx], scores[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def fit_and_score(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = 42
) -> tuple[dict[str, Pipeline], dict[str, np.ndarray], pd.DataFrame]:
    """Fit every candidate on ``train`` and return fitted models, test scores and a comparison table."""
    X_train, y_train = _prepare(train), train[TARGET].to_numpy()
    X_test, y_test = _prepare(test), test[TARGET].to_numpy()
    fitted, scores, rows = {}, {}, []
    for name, pipeline in candidate_models(seed).items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        scores[name] = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, scores[name])
        low, high = bootstrap_ci(y_test, scores[name], test["employee_id"].to_numpy(), seed=seed)
        rows.append({"model": name, **metrics, "roc_auc_ci_low": low, "roc_auc_ci_high": high})
    return fitted, scores, pd.DataFrame(rows).set_index("model")


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Standardised logistic-regression coefficients as odds ratios, largest effect first."""
    names = model.named_steps["prep"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coefficient": coefficients})
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
