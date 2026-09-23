"""L1 logistic regression over per-policy Jev probabilities, tuned for F0.5."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import scan_best_threshold_f05

DEFAULT_CS = [float(c) for c in np.logspace(-3, 1, 17)]
DEFAULT_CLASS_WEIGHTS = [None, "balanced"]


def make_model(C: float, class_weight: str | None) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(l1_ratio=1, solver="liblinear", C=C, class_weight=class_weight, max_iter=2000),
    )


def fit_l1_logistic(
    X: np.ndarray,
    y: np.ndarray,
    Cs: list[float] = DEFAULT_CS,
    class_weights: list[str | None] = DEFAULT_CLASS_WEIGHTS,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Grid-search (C, class_weight) by out-of-fold F0.5, with the threshold tuned on the same OOF probabilities.

    Returns the grid results, the best config, its OOF probabilities, and a model refit on all of X.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    grid = []
    best = None
    for class_weight in class_weights:
        for C in Cs:
            model = make_model(C, class_weight)
            oof_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            threshold, _ = scan_best_threshold_f05(y.tolist(), oof_prob.tolist())
            n_selected = int(np.count_nonzero(model.fit(X, y)[-1].coef_))
            f05 = _f05(y, oof_prob, threshold)
            entry = {"C": C, "class_weight": class_weight or "none", "threshold": threshold,
                     "oof_f0.5": f05, "n_selected": n_selected}
            grid.append(entry)
            if best is None or f05 > best["entry"]["oof_f0.5"]:
                best = {"entry": entry, "oof_prob": oof_prob, "class_weight": class_weight}

    final = make_model(best["entry"]["C"], best["class_weight"]).fit(X, y)
    return {"grid": grid, "best": best["entry"], "oof_prob": best["oof_prob"], "model": final}


def selected_positions(model: Pipeline) -> list[int]:
    return [int(j) for j in np.flatnonzero(model[-1].coef_[0])]


def predict_from_selected(model: Pipeline, X_selected: np.ndarray, positions: list[int]) -> np.ndarray:
    """Predict with only the selected columns; unselected columns have zero weight, so fill them with the training mean."""
    X = np.tile(model[0].mean_, (X_selected.shape[0], 1))
    X[:, positions] = X_selected
    return model.predict_proba(X)[:, 1]


def _f05(y: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    from sklearn.metrics import fbeta_score
    return float(fbeta_score(y, (prob >= threshold).astype(int), beta=0.5, zero_division=0))
