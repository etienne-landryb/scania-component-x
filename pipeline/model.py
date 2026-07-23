"""
model.py -- Phase 3 modelling for SCANIA Component X.

LightGBM multiclass model over the Phase 2 features, plus the evaluation
protocol that turns predictions into the challenge's Total_cost.

WHY LIGHTGBM
------------
Mixed numeric + categorical features, missing values, 187k x 415 -- gradient
boosting handles all of it natively, trains in minutes on a laptop, and gives
feature importances that make the result explainable. Deep models were the
published state of the art here; matching them is explicitly not the goal.

CALIBRATION MATTERS
-------------------
The cost-minimising decision rule (costs.py) consumes PROBABILITIES. If those
probabilities are badly calibrated the rule mis-escalates, so we:
  * do NOT resample or reweight by default (that distorts calibration), and
  * measure calibration explicitly, and
  * report the cost-rule result against argmax so the comparison is honest,
    rather than assuming the cost layer must win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from costs import (COST_MATRIX, total_cost, cost_breakdown,
                   expected_cost_decision, accuracy_optimal_decision,
                   naive_all_zero)

ID_COL = "vehicle_id"


def split_features(X: pd.DataFrame):
    """Drop the identifier; return (features, categorical column names).

    vehicle_id must never be a feature -- it carries no signal and would let
    the model memorise individual trucks.
    """
    feats = X.drop(columns=[ID_COL], errors="ignore")
    cat_cols = [c for c in feats.columns if str(feats[c].dtype) == "category"]
    return feats, cat_cols


def make_cost_eval():
    """LightGBM eval metric = Total_cost after the expected-cost decision rule.

    WHY THIS EXISTS: with ~97% class 0, multi_logloss is minimised almost
    immediately by predicting the base rate, so early stopping fires after a
    handful of rounds and the model never learns the rare classes. Early
    stopping must watch the metric we actually care about.
    """
    def cost_eval(y_true, y_pred):
        proba = np.asarray(y_pred)
        if proba.ndim == 1:                       # some versions flatten
            proba = proba.reshape(len(y_true), -1, order="F")
        pred = expected_cost_decision(proba)
        return "total_cost", total_cost(y_true, pred), False   # lower is better
    return cost_eval


def train_model(X_train: pd.DataFrame, y_train: np.ndarray,
                X_valid: pd.DataFrame | None = None,
                y_valid: np.ndarray | None = None,
                class_weighted: bool = False,
                n_estimators: int = 1500, learning_rate: float = 0.05,
                num_leaves: int = 63, seed: int = 42, verbose: bool = True,
                early_stopping_rounds: int = 150):
    """Fit a LightGBM multiclass model.

    class_weighted=False (default) keeps probabilities calibrated and lets the
    cost matrix do the work at decision time. Setting it True weights samples
    by the average cost of misclassifying their true class -- a variant worth
    comparing, not an automatic improvement.
    """
    import lightgbm as lgb

    Xt, cat_cols = split_features(X_train)

    sample_weight = None
    if class_weighted:
        # average off-diagonal cost of getting each true class wrong
        row_cost = COST_MATRIX.sum(axis=1) / (COST_MATRIX.shape[1] - 1)
        sample_weight = row_cost[y_train.astype(int)]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=COST_MATRIX.shape[0],
        # Switch OFF the built-in multi_logloss. Otherwise LightGBM evaluates it
        # first and early stopping latches onto it (stopping at ~iteration 4 under
        # this class imbalance) instead of our Total_cost metric.
        metric="None",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=40,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    fit_kw = {}
    if X_valid is not None and y_valid is not None:
        Xv, _ = split_features(X_valid)
        fit_kw["eval_set"] = [(Xv, y_valid)]
        # Early-stop on Total_cost, NOT on logloss (see make_cost_eval).
        fit_kw["eval_metric"] = make_cost_eval()
        callbacks = [lgb.early_stopping(early_stopping_rounds,
                                        first_metric_only=True, verbose=False)]
        if verbose:
            callbacks.append(lgb.log_evaluation(100))
        fit_kw["callbacks"] = callbacks

    model.fit(Xt, y_train, sample_weight=sample_weight,
              categorical_feature=cat_cols or "auto", **fit_kw)
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    feats, _ = split_features(X)
    return model.predict_proba(feats)


def evaluate_policies(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Compare the three decision policies on the official metric."""
    n = len(y_true)
    rows = []
    for name, pred in [
        ("Naive (all healthy)", naive_all_zero(n)),
        ("Accuracy-optimal (argmax)", accuracy_optimal_decision(proba)),
        ("Cost-minimising (expected cost)", expected_cost_decision(proba)),
    ]:
        acc = float((pred == y_true).mean())
        rows.append({
            "policy": name,
            "total_cost": total_cost(y_true, pred),
            "cost_per_vehicle": total_cost(y_true, pred) / n,
            "accuracy": round(acc, 4),
            "n_flagged": int((pred > 0).sum()),
        })
    return pd.DataFrame(rows)


def calibration_report(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Per-class: mean predicted probability vs actual frequency.

    If these two columns diverge badly, the cost rule is standing on sand --
    which is exactly what we want to know before trusting it.
    """
    rows = []
    for c in range(proba.shape[1]):
        rows.append({
            "class": c,
            "mean_predicted_p": round(float(proba[:, c].mean()), 5),
            "actual_frequency": round(float((y_true == c).mean()), 5),
            "n_actual": int((y_true == c).sum()),
        })
    return pd.DataFrame(rows)


def top_features(model, X: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    feats, _ = split_features(X)
    imp = pd.DataFrame({
        "feature": feats.columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    })
    return imp.sort_values("gain", ascending=False).head(k).reset_index(drop=True)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    n = COST_MATRIX.shape[0]
    m = np.zeros((n, n), dtype=int)
    for a, p in zip(y_true.astype(int), y_pred.astype(int)):
        m[a, p] += 1
    return pd.DataFrame(m,
                        index=[f"actual_{i}" for i in range(n)],
                        columns=[f"pred_{i}" for i in range(n)])
