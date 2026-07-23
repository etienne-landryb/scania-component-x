"""
calibrate.py -- Phase 3.5: probability calibration.

WHY THIS EXISTS
---------------
The cost-minimising rule in costs.py consumes PROBABILITIES:

    E[cost | action m] = sum_n  p_n * COST[n][m]

so the quality of p matters as much as the quality of the ranking. On the raw
model, class 1 was over-predicted ~8x and class 4 under-predicted ~1.85x
against their true frequencies. Every one of those distortions is fed straight
into the decision rule, which is why it flagged ~52% of the fleet.

Calibration is therefore not cosmetic here -- it is the layer the whole
cost argument rests on.

FITTED ON HELD-OUT TRAINING VEHICLES, never on validation or test. The
validation set has only 16 class-1 and 76 class-4 vehicles; fitting a
calibrator on that would overfit immediately and quietly flatter the result.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


class ProbabilityCalibrator:
    """Post-hoc multiclass calibration.

    method='dirichlet'  multinomial logistic regression on log-probabilities.
                        Low variance, corrects systematic per-class bias, and
                        safe when the rare classes have few examples.
    method='isotonic'   per-class one-vs-rest isotonic regression, renormalised.
                        More flexible, needs more minority examples.
    method='none'       passthrough (used as the control in comparisons).
    """

    def __init__(self, method: str = "dirichlet"):
        if method not in ("dirichlet", "isotonic", "none"):
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method
        self.model = None
        self.calibrators: list = []
        self.n_classes = None

    def fit(self, proba: np.ndarray, y: np.ndarray):
        proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
        y = np.asarray(y, dtype=int)
        self.n_classes = proba.shape[1]

        if self.method == "none":
            return self

        if self.method == "dirichlet":
            from sklearn.linear_model import LogisticRegression
            # NB: no multi_class= argument -- it was removed in recent
            # scikit-learn; multinomial is the default for multiclass targets.
            self.model = LogisticRegression(max_iter=2000, C=1.0)
            self.model.fit(np.log(proba), y)
            # remember which classes were actually present during fitting
            self._fitted_classes = self.model.classes_
            return self

        from sklearn.isotonic import IsotonicRegression
        self.calibrators = []
        for c in range(self.n_classes):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(proba[:, c], (y == c).astype(float))
            self.calibrators.append(iso)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)

        if self.method == "none":
            return proba / proba.sum(axis=1, keepdims=True)

        if self.method == "dirichlet":
            out = self.model.predict_proba(np.log(proba))
            # restore full class layout if a class was absent when fitting
            if out.shape[1] != self.n_classes:
                full = np.zeros((len(proba), self.n_classes))
                for j, c in enumerate(self._fitted_classes):
                    full[:, int(c)] = out[:, j]
                out = full
            return np.clip(out, EPS, 1.0) / np.clip(out, EPS, 1.0).sum(axis=1, keepdims=True)

        cols = [self.calibrators[c].predict(proba[:, c]) for c in range(self.n_classes)]
        out = np.clip(np.column_stack(cols), EPS, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def fit_transform(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(proba, y).transform(proba)


def calibration_error(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean absolute gap between mean predicted probability and true frequency.

    A single number to compare calibrators. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    gaps = []
    for c in range(proba.shape[1]):
        gaps.append(abs(proba[:, c].mean() - float((y_true == c).mean())))
    return float(np.mean(gaps))


def split_by_vehicle(X, y, vehicle_col: str = "vehicle_id",
                     holdout_frac: float = 0.2, seed: int = 42):
    """Split rows into (fit, calibration) parts WITHOUT splitting a vehicle.

    Cut points from one truck must not appear on both sides -- that would leak
    a vehicle's own history into its calibration data.
    """
    vids = X[vehicle_col].to_numpy()
    unique = np.unique(vids)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(unique, size=int(len(unique) * holdout_frac),
                          replace=False).tolist())
    mask_cal = np.fromiter((v in held for v in vids), dtype=bool, count=len(vids))
    return (X[~mask_cal], y[~mask_cal]), (X[mask_cal], y[mask_cal])


class PriorShiftCalibrator:
    """Correct for LABEL SHIFT between training and deployment class priors.

    Discovered empirically on this dataset: uniform cut-point sampling gives a
    training prior of roughly

        class 1 = 1.40%,  class 4 = 0.48%

    while the challenge's last-readout selection gives

        class 1 = 0.32%,  class 4 = 1.51%

    -- the two rarest classes are essentially inverted. A model fitted on the
    first prior systematically over-predicts class 1 and under-predicts class 4,
    which is exactly the miscalibration observed, and it feeds straight into the
    cost rule (class 4 is the expensive one to miss).

    The standard correction under label shift keeps the likelihood and swaps the
    prior:

        p_corrected(c | x)  proportional to  p_model(c | x) * pi_target(c) / pi_train(c)

    Only the 5 class priors are estimated, so this is a 5-parameter adjustment
    rather than a fitted model -- far less prone to overfitting than isotonic on
    a handful of minority examples.

    NOTE ON HONESTY: pi_target is estimated from validation labels. That is a
    property of the evaluation protocol (how last readouts were chosen), not of
    the test vehicles, and it is reported as a modelling choice rather than
    hidden.
    """

    def __init__(self, train_prior, target_prior):
        self.train_prior = np.clip(np.asarray(train_prior, dtype=float), EPS, None)
        self.target_prior = np.clip(np.asarray(target_prior, dtype=float), EPS, None)
        self.ratio = self.target_prior / self.train_prior

    @classmethod
    def from_labels(cls, y_train, y_target, n_classes: int = 5):
        def prior(y):
            y = np.asarray(y, dtype=int)
            return np.array([(y == c).mean() for c in range(n_classes)])
        return cls(prior(y_train), prior(y_target))

    def fit(self, proba=None, y=None):
        return self                      # priors already supplied

    def transform(self, proba: np.ndarray) -> np.ndarray:
        out = np.clip(np.asarray(proba, dtype=float), EPS, 1.0) * self.ratio
        return out / out.sum(axis=1, keepdims=True)

    def describe(self):
        return {
            "train_prior": np.round(self.train_prior, 5).tolist(),
            "target_prior": np.round(self.target_prior, 5).tolist(),
            "adjustment_ratio": np.round(self.ratio, 3).tolist(),
        }
