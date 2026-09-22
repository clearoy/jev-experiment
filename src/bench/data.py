"""Loading rows from the vcbench founder-success CSVs."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PUBLIC_DATA_FILE = DATA_DIR / "vcbench_final_public.csv"
PRIVATE_DATA_FILE = DATA_DIR / "vcbench_final_private.csv"


def load_rows(data_file: Path, limit: int | None = None) -> list[dict]:
    with open(data_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows
