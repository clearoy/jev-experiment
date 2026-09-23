"""Policy-assisted Jev: one noul per (founder, policy), combined with L1 logistic regression.

1. For every PUBLIC founder x policy, ask Jev whether the founder will succeed,
   with the policy given as guidance. Policies for one founder are batched into
   a single request; each is still answered independently.
2. Fit an L1 logistic regression on the per-policy probabilities, choosing C,
   class_weight and the decision threshold by 5-fold out-of-fold F0.5.
3. Run only the selected (nonzero-weight) policies on the PRIVATE founders and
   evaluate the fitted model + threshold there.

Judgements are cached under result/jev_with_policies/cache/ and reused across
runs, so reruns resume and extending to more policies only asks the new ones.
Everything else goes to result/jev_with_policies/<timestamp>_p<N>/.

Usage:
    python script/jev_with_policies.py --policy-limit 30
    python script/jev_with_policies.py                    # all 201 policies
"""

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from bench import (  # noqa: E402
    POLICIES_FILE,
    POLICY_INSTRUCTIONS_TEMPLATE,
    PRIVATE_DATA_FILE,
    PUBLIC_DATA_FILE,
    RESULT_DIR,
    build_policy_matrix,
    compute_metrics,
    fit_l1_logistic,
    load_policies,
    load_rows,
    predict_from_selected,
    read_judgements,
    run_policy_predictions,
    scan_best_threshold_f05,
    selected_positions,
)

BASE_DIR = RESULT_DIR / "jev_with_policies"
CACHE_DIR = BASE_DIR / "cache"
PUBLIC_CACHE = CACHE_DIR / "public_judgements.csv"
PRIVATE_CACHE = CACHE_DIR / "private_judgements.csv"


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, rows: list[dict], prob: np.ndarray, threshold: float) -> None:
    write_csv(path, ["founder_uuid", "success", "probability", "predicted"], [
        {"founder_uuid": r["founder_uuid"], "success": int(r["success"]),
         "probability": float(p), "predicted": int(p >= threshold)}
        for r, p in zip(rows, prob)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-limit", type=int, default=None, help="only use the first N policies")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N founders of each split")
    parser.add_argument("--concurrency", type=int, default=20, help="max concurrent requests")
    parser.add_argument("--chunk-size", type=int, default=50, help="max policy questions per request")
    args = parser.parse_args()

    policies = load_policies(POLICIES_FILE, args.policy_limit)
    policy_idxs = [p["idx"] for p in policies]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = BASE_DIR / f"{timestamp}_p{len(policies)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Public judgements.
    public_rows = load_rows(PUBLIC_DATA_FILE, args.limit)
    print(f"[public] {len(public_rows)} founders x {len(policies)} policies")
    asyncio.run(run_policy_predictions(public_rows, policies, PUBLIC_CACHE, args.concurrency, args.chunk_size))

    X_public = np.array(build_policy_matrix(read_judgements(PUBLIC_CACHE), public_rows, policy_idxs))
    y_public = np.array([int(r["success"]) for r in public_rows])
    write_csv(run_dir / "public_policy_matrix.csv", ["founder_uuid", "success", *[f"p{i}" for i in policy_idxs]], [
        {"founder_uuid": r["founder_uuid"], "success": int(r["success"]),
         **{f"p{i}": v for i, v in zip(policy_idxs, x)}}
        for r, x in zip(public_rows, X_public)
    ])

    # 2. Fit L1 logistic regression.
    print("[fit] grid-searching C and class_weight by out-of-fold F0.5")
    fit = fit_l1_logistic(X_public, y_public)
    best, model, oof_prob = fit["best"], fit["model"], fit["oof_prob"]
    threshold = best["threshold"]
    positions = selected_positions(model)
    coefs = model[-1].coef_[0]
    selected = [{"idx": policy_idxs[j], "coef": float(coefs[j]), "policy": policies[j]["policy"]} for j in positions]
    selected.sort(key=lambda s: -abs(s["coef"]))

    write_csv(run_dir / "cv_grid.csv", ["C", "class_weight", "threshold", "oof_f0.5", "n_selected"], fit["grid"])
    write_csv(run_dir / "selected_policies.csv", ["idx", "coef", "policy"], selected)
    write_predictions(run_dir / "public_oof_predictions.csv", public_rows, oof_prob, threshold)
    _, scan = scan_best_threshold_f05(y_public.tolist(), oof_prob.tolist())
    write_csv(run_dir / "public_oof_threshold_scan.csv", ["threshold", "f0.5"], scan)
    joblib.dump(model, run_dir / "model.joblib")

    public_oof_metrics = compute_metrics(y_public.tolist(), oof_prob.tolist(), threshold)
    public_insample_prob = model.predict_proba(X_public)[:, 1]
    public_insample_metrics = compute_metrics(y_public.tolist(), public_insample_prob.tolist(), threshold)
    write_json(run_dir / "public_metrics.json", {
        "out_of_fold": public_oof_metrics,
        "in_sample_refit": public_insample_metrics,
    })
    print(f"[fit] best C={best['C']:.4g}, class_weight={best['class_weight']}, threshold={threshold:.4f}, "
          f"OOF F0.5={best['oof_f0.5']:.4f}, {len(selected)}/{len(policies)} policies selected")

    if not selected:
        print("No policies selected, so there is nothing to run on the private set.")
        return

    # 3. Private judgements for the selected policies only, then evaluate.
    private_rows = load_rows(PRIVATE_DATA_FILE, args.limit)
    selected_policies = [policies[j] for j in positions]
    print(f"[private] {len(private_rows)} founders x {len(selected_policies)} selected policies")
    asyncio.run(run_policy_predictions(private_rows, selected_policies, PRIVATE_CACHE, args.concurrency, args.chunk_size))

    X_private_sel = np.array(build_policy_matrix(
        read_judgements(PRIVATE_CACHE), private_rows, [p["idx"] for p in selected_policies],
    ))
    y_private = [int(r["success"]) for r in private_rows]
    private_prob = predict_from_selected(model, X_private_sel, positions)
    private_metrics = compute_metrics(y_private, private_prob.tolist(), threshold)
    write_predictions(run_dir / "private_predictions.csv", private_rows, private_prob, threshold)
    write_json(run_dir / "private_metrics.json", private_metrics)

    write_json(run_dir / "summary.json", {
        "timestamp": timestamp,
        "instructions_template": POLICY_INSTRUCTIONS_TEMPLATE,
        "n_policies": len(policies),
        "policy_idxs": policy_idxs,
        "n_public": len(public_rows),
        "n_private": len(private_rows),
        "public_cache": str(PUBLIC_CACHE.relative_to(ROOT)),
        "private_cache": str(PRIVATE_CACHE.relative_to(ROOT)),
        "best_config": best,
        "selected_policy_idxs": [s["idx"] for s in selected],
        "public_oof_metrics": public_oof_metrics,
        "private_metrics": private_metrics,
    })

    print(json.dumps({"public_oof": public_oof_metrics, "private": private_metrics}, indent=2))
    print(f"\nSaved everything to {run_dir}")


if __name__ == "__main__":
    main()
