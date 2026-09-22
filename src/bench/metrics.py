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
    from sklearn.metrics import fbeta_score

    candidates = sorted(set(y_prob) | {0.0})
    table = []
    for t in candidates:
        y_pred = [1 if p >= t else 0 for p in y_prob]
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        table.append({"threshold": t, "f0.5": f05})

    best = max(table, key=lambda r: r["f0.5"])
    return best["threshold"], table
