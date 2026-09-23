"""Scoring predictions against the success label."""


def compute_metrics(y_true: list[int], y_prob: list[float], threshold: float) -> dict:
    from sklearn.metrics import (
        confusion_matrix,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": len(y_true),
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1.0, zero_division=0),
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def scan_best_threshold_f05(y_true: list[int], y_prob: list[float]) -> tuple[float, list[dict]]:
    """Scan candidate thresholds (every distinct probability) and return the one maximizing F0.5."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    denom = 0.25 * precision + recall
    f05 = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    table = [{"threshold": float(t), "f0.5": float(f)} for t, f in zip(thresholds, f05)]
    best = int(np.argmax(f05))
    return float(thresholds[best]), table
