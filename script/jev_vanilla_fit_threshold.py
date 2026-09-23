"""Fit a decision threshold for jev's noul success prediction, using F0.5.

1. Runs the same noul question as jev_vanilla.py, but against the PUBLIC
   dataset (4500 rows -> 4500 jev calls).
2. Scans candidate thresholds on those public probabilities and picks the one
   that maximizes F0.5.
3. Re-evaluates the most recent existing jev_vanilla.py PRIVATE run (reusing
   its already-computed probabilities, no extra jev calls) at that fitted
   threshold.

All outputs are saved under result/.

Usage:
    python script/jev_vanilla_fit_threshold.py [--limit N] [--concurrency N]
"""

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from bench import (  # noqa: E402
    VANILLA_INSTRUCTIONS,
    PUBLIC_DATA_FILE,
    RESULT_DIR,
    compute_metrics,
    latest_result_file,
    load_rows,
    read_details_csv,
    run_predictions,
    scan_best_threshold_f05,
    write_details_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only use the first N public rows")
    parser.add_argument("--concurrency", type=int, default=20, help="max concurrent requests")
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. Run jev on the public dataset.
    rows = load_rows(PUBLIC_DATA_FILE, args.limit)
    print(f"Running jev noul on {len(rows)} rows from {PUBLIC_DATA_FILE.name} (concurrency={args.concurrency})")
    public_results = asyncio.run(run_predictions(rows, args.concurrency))
    public_y_true = [r["success"] for r in public_results]
    public_y_prob = [r["probability"] for r in public_results]

    public_details_path = RESULT_DIR / f"jev_fit_threshold_public_details_{timestamp}.csv"
    write_details_csv(public_details_path, public_results, threshold=0.5)
    print(f"Saved public details to {public_details_path}")

    # 2. Scan thresholds on the public run, pick the one maximizing F0.5.
    best_threshold, scan_table = scan_best_threshold_f05(public_y_true, public_y_prob)
    scan_path = RESULT_DIR / f"jev_fit_threshold_scan_{timestamp}.csv"
    with open(scan_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "f0.5"])
        writer.writeheader()
        writer.writerows(scan_table)
    print(f"Saved threshold scan to {scan_path}")

    public_metrics_at_best = compute_metrics(public_y_true, public_y_prob, best_threshold)
    summary = {
        "instructions": VANILLA_INSTRUCTIONS,
        "dataset": PUBLIC_DATA_FILE.name,
        "timestamp": timestamp,
        "best_threshold": best_threshold,
        "public_metrics_at_best_threshold": public_metrics_at_best,
    }
    summary_path = RESULT_DIR / f"jev_fit_threshold_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved fit summary to {summary_path}")

    # 3. Re-evaluate the existing private run's probabilities at the fitted threshold (no new jev calls).
    private_details_path = latest_result_file("jev_vanilla_details_*.csv")
    if private_details_path is None:
        print("\nNo existing jev_vanilla.py private result found under result/ — skipping re-evaluation.")
        return

    private_y_true, private_y_prob = read_details_csv(private_details_path)
    private_metrics_at_best = compute_metrics(private_y_true, private_y_prob, best_threshold)
    eval_out = {
        "instructions": VANILLA_INSTRUCTIONS,
        "dataset": "vcbench_final_private.csv",
        "reused_details_file": private_details_path.name,
        "fitted_threshold_source": "vcbench_final_public.csv",
        "timestamp": timestamp,
        "best_threshold": best_threshold,
        **private_metrics_at_best,
    }
    eval_path = RESULT_DIR / f"jev_fit_threshold_private_eval_{timestamp}.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2)

    print(f"\nRe-evaluated {private_details_path.name} at threshold={best_threshold}:")
    print(json.dumps(eval_out, indent=2))
    print(f"Saved private re-evaluation to {eval_path}")


if __name__ == "__main__":
    main()
