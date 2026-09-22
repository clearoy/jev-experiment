"""Smoke test: run Jev's three question primitives against one row of the public dataset.

Usage:
    python script/test_jev.py
"""

import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from jev import JevClient  # noqa: E402

DATA_FILE = ROOT / "data" / "vcbench_final_public.csv"


def load_sample_row() -> dict:
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader)


def main() -> None:
    row = load_sample_row()
    state = {
        "industry": row["industry"],
        "prose": row["anonymised_prose"][:2000],
    }

    with JevClient() as client:
        noul_answer = client.noul(
            state,
            instructions="Does this founder have prior startup experience?",
        )
        print(f"noul (prior startup experience): {noul_answer}")

        choice_answer = client.choice(
            state,
            instructions="What is this founder's most likely background?",
            criteria={
                "technical": "Engineering, product, or other hands-on technical background",
                "business": "Sales, marketing, finance, or general management background",
                "academic": "Primarily research or academic background",
            },
        )
        print(f"choice (background): {choice_answer.choice} ({choice_answer.probabilities})")

        score_answer = client.score(
            state,
            instructions="How strong is this founder's track record?",
            criteria=["weak", "average", "strong"],
        )
        print(f"score (track record): {score_answer.score} (confidence={score_answer.confidence})")


if __name__ == "__main__":
    main()
