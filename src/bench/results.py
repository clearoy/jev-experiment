"""Reading and writing files under result/."""

import csv
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent.parent.parent / "result"


def write_details_csv(path: Path, results: list[dict], threshold: float) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["founder_uuid", "success", "probability", "predicted"])
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "predicted": int(r["probability"] >= threshold)})


def read_details_csv(path: Path) -> tuple[list[int], list[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y_true = [int(r["success"]) for r in rows]
    y_prob = [float(r["probability"]) for r in rows]
    return y_true, y_prob


def latest_result_file(pattern: str) -> Path | None:
    matches = sorted(RESULT_DIR.glob(pattern))
    return matches[-1] if matches else None
