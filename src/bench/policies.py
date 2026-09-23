"""Loading investor policies and turning them into Jev noul questions."""

import csv
from pathlib import Path

from jev import Noul

from ._prompts import POLICY_INSTRUCTIONS_TEMPLATE
from .data import DATA_DIR

POLICIES_FILE = DATA_DIR / "policies" / "policies.csv"


def load_policies(path: Path = POLICIES_FILE, limit: int | None = None) -> list[dict]:
    """Return [{idx, policy}] in file order, optionally only the first `limit`."""
    with open(path, newline="", encoding="utf-8") as f:
        policies = [{"idx": int(r["idx"]), "policy": r["policy"]} for r in csv.DictReader(f)]
    return policies[:limit] if limit else policies


def policy_question(policy: str) -> Noul:
    return Noul(instructions=POLICY_INSTRUCTIONS_TEMPLATE.format(policy=policy))
