"""Vanilla Jev baseline: predict founder success from anonymised_prose alone.

Runs a single Noul question ("will this founder be successful?") against every
row of the private dataset, then scores the predictions against the `success`
label. Saves per-row predictions and summary metrics under result/.

Usage:
    python script/jev_vanilla.py [--limit N] [--concurrency N] [--threshold F]
"""

import argparse
import asyncio
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
    PRIVATE_DATA_FILE,
    RESULT_DIR,
    compute_metrics,
    load_rows,
    run_predictions,
    write_details_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only use the first N rows")
    parser.add_argument("--concurrency", type=int, default=20, help="max concurrent requests")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold on the noul probability")
    args = parser.parse_args()

    rows = load_rows(PRIVATE_DATA_FILE, args.limit)
    print(f"Running jev noul on {len(rows)} rows from {PRIVATE_DATA_FILE.name} (concurrency={args.concurrency})")

    results = asyncio.run(run_predictions(rows, args.concurrency))
    y_true = [r["success"] for r in results]
    y_prob = [r["probability"] for r in results]
    metrics = compute_metrics(y_true, y_prob, args.threshold)

    RESULT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    details_path = RESULT_DIR / f"jev_vanilla_details_{timestamp}.csv"
    metrics_path = RESULT_DIR / f"jev_vanilla_metrics_{timestamp}.json"

    write_details_csv(details_path, results, args.threshold)

    metrics_out = {
        "instructions": VANILLA_INSTRUCTIONS,
        "dataset": PRIVATE_DATA_FILE.name,
        "timestamp": timestamp,
        **metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(json.dumps(metrics_out, indent=2))
    print(f"\nSaved details to {details_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
