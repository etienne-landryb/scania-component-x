"""
costs.py -- the decision-theoretic core of the SCANIA Component X project.

The challenge is NOT scored on accuracy. It is scored on Total_cost under an
asymmetric cost matrix: missing an imminent failure costs up to 500, while a
false alarm costs 7-10. Optimising accuracy therefore optimises the wrong thing.

ORIENTATION (easy to get backwards, so it is stated and tested explicitly):

    COST[n][m]  where  n = ACTUAL class, m = PREDICTED class

    row n=4, col m=0  -> 500  (truck about to fail, called healthy: worst case)
    row n=0, col m=4  ->  10  (healthy truck, urgent alarm: cheap mistake)
    diagonal          ->   0  (correct prediction costs nothing)

DECISION RULE
-------------
Given predicted class probabilities p = (p_0..p_4) for a vehicle, the expected
cost of *acting as if* the class were m is:

    E[cost | m] = sum_n  p_n * COST[n][m]

We pick the m that minimises it. This is standard Bayes decision theory: the
model estimates probabilities, and the cost matrix -- not the probabilities --
decides the action. It is why a vehicle with only a 10% chance of imminent
failure can still be worth flagging: 0.10 * 500 outweighs a cost-10 false alarm.

This is also why we do NOT resample the training data to "fix" the imbalance:
resampling distorts the probabilities that this rule depends on. The imbalance
is handled here, at the decision layer, where it belongs.
"""

from __future__ import annotations

import numpy as np

# Rows = actual (0..4), columns = predicted (0..4). From the challenge spec.
COST_MATRIX = np.array([
    [0,   7,   8,   9,  10],    # actual 0 (healthy)  -> unnecessary workshop check
    [200, 0,   7,   8,   9],    # actual 1
    [300, 200, 0,   7,   8],    # actual 2
    [400, 300, 200, 0,   7],    # actual 3
    [500, 400, 300, 200, 0],    # actual 4 (imminent) -> missed failure
], dtype=float)

N_CLASSES = COST_MATRIX.shape[0]

CLASS_MEANING = {
    0: "No failure expected (>48 time units, or healthy)",
    1: "Failure in 24-48 time units",
    2: "Failure in 12-24 time units",
    3: "Failure in 6-12 time units",
    4: "Imminent failure (0-6 time units)",
}

# Plain-language action implied by each predicted class.
CLASS_ACTION = {
    0: "No action - keep monitoring",
    1: "Plan an inspection at the next scheduled service",
    2: "Schedule an inspection soon",
    3: "Book a workshop visit promptly",
    4: "Urgent: take the vehicle in before the next long mission",
}


def total_cost(y_true, y_pred, cost_matrix: np.ndarray = COST_MATRIX) -> float:
    """Official metric: sum of COST[actual, predicted] over all vehicles."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(cost_matrix[y_true, y_pred].sum())


def cost_breakdown(y_true, y_pred, cost_matrix: np.ndarray = COST_MATRIX) -> dict:
    """Where the cost comes from: per-actual-class totals and counts."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {}
    for n in range(cost_matrix.shape[0]):
        m = y_true == n
        if not m.any():
            continue
        out[n] = {
            "n_vehicles": int(m.sum()),
            "cost": float(cost_matrix[y_true[m], y_pred[m]].sum()),
            "n_correct": int((y_pred[m] == n).sum()),
        }
    return out


def expected_cost_decision(proba: np.ndarray,
                           cost_matrix: np.ndarray = COST_MATRIX) -> np.ndarray:
    """Pick, per row, the class that minimises expected cost.

    proba: (n_samples, n_classes) predicted probabilities.
    Returns integer predictions of shape (n_samples,).

    expected[i, m] = sum_n proba[i, n] * cost_matrix[n, m]   ->  proba @ cost_matrix
    """
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] != cost_matrix.shape[0]:
        raise ValueError(
            f"proba has {proba.shape[1]} classes but cost matrix has {cost_matrix.shape[0]}"
        )
    expected = proba @ cost_matrix          # (n_samples, n_predicted_classes)
    return expected.argmin(axis=1)


def expected_costs_per_action(proba_row: np.ndarray,
                              cost_matrix: np.ndarray = COST_MATRIX) -> np.ndarray:
    """Expected cost of each possible action for ONE vehicle (for the app's
    explanation panel: 'here is why this action was chosen')."""
    return np.asarray(proba_row, dtype=float) @ cost_matrix


def naive_all_zero(n_samples: int) -> np.ndarray:
    """Baseline: assume every truck is healthy. The 'do nothing' policy."""
    return np.zeros(n_samples, dtype=int)


def accuracy_optimal_decision(proba: np.ndarray) -> np.ndarray:
    """Baseline: predict the most likely class, ignoring cost."""
    return np.asarray(proba).argmax(axis=1)
