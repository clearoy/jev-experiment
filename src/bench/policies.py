"""Loading investor policies and turning them into Jev noul questions."""

import csv
from pathlib import Path

from jev import Noul

from .data import DATA_DIR

POLICIES_FILE = DATA_DIR / "policies" / "policies.csv"

POLICY_INSTRUCTIONS_TEMPLATE = (
    "Investor heuristic (guidance, not a strict rule): {policy}\n"
    "Considering this heuristic along with the founder's full profile, "
    "will this founder be successful?"
)


def load_policies(path: Path = POLICIES_FILE, limit: int | None = None) -> list[dict]:
    """Return [{idx, policy}] in file order, optionally only the first `limit`."""
    with open(path, newline="", encoding="utf-8") as f:
        policies = [{"idx": int(r["idx"]), "policy": r["policy"]} for r in csv.DictReader(f)]
    return policies[:limit] if limit else policies


def policy_question(policy: str) -> Noul:
    return Noul(instructions=POLICY_INSTRUCTIONS_TEMPLATE.format(policy=policy))
