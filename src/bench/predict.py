"""Run jev's noul primitive over dataset rows to predict founder success."""

import asyncio
import csv
from pathlib import Path

from jev import AsyncJevClient

from .policies import policy_question
from .results import JUDGEMENT_FIELDS, read_judgements

INSTRUCTIONS = (
    "Will this startup founder be successful, based on their educational "
    "background, professional experience, and industry?"
)


async def run_predictions(
    rows: list[dict],
    concurrency: int = 20,
    instructions: str = INSTRUCTIONS,
) -> list[dict]:
    """Call noul(anonymised_prose) for every row, concurrently.

    Returns a list of {founder_uuid, success, probability} dicts, in input order.
    """
    results: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async with AsyncJevClient() as client:
        async def worker(i: int, row: dict) -> None:
            nonlocal done
            async with sem:
                probability = await client.noul(row["anonymised_prose"], instructions=instructions)
            results[i] = {
                "founder_uuid": row["founder_uuid"],
                "success": int(row["success"]),
                "probability": probability,
            }
            done += 1
            if done % 100 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}", flush=True)

        await asyncio.gather(*(worker(i, row) for i, row in enumerate(rows)))

    return results


async def run_policy_predictions(
    rows: list[dict],
    policies: list[dict],
    cache_path: Path,
    concurrency: int = 20,
    chunk_size: int = 50,
) -> None:
    """Ask one noul per (founder, policy) and append each judgement to `cache_path`.

    Policies for the same founder are batched into one request (up to `chunk_size`
    questions); Jev answers each question independently. Pairs already in the
    cache are skipped, so an interrupted run resumes where it stopped.
    """
    done = set(read_judgements(cache_path)) if cache_path.exists() else set()

    jobs = []
    for row in rows:
        missing = [p for p in policies if (row["founder_uuid"], p["idx"]) not in done]
        for start in range(0, len(missing), chunk_size):
            jobs.append((row, missing[start:start + chunk_size]))

    total_pairs = len(rows) * len(policies)
    print(f"  {total_pairs - sum(len(c) for _, c in jobs)}/{total_pairs} judgements cached, "
          f"{len(jobs)} requests to send", flush=True)
    if not jobs:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not cache_path.exists()
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    with open(cache_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGEMENT_FIELDS)
        if is_new:
            writer.writeheader()

        async with AsyncJevClient() as client:
            async def worker(row: dict, chunk: list[dict]) -> None:
                nonlocal completed
                questions = {f"p{p['idx']}": policy_question(p["policy"]) for p in chunk}
                async with sem:
                    response = await client.ask(row["anonymised_prose"], questions)
                for p in chunk:
                    writer.writerow({
                        "founder_uuid": row["founder_uuid"],
                        "success": int(row["success"]),
                        "policy_idx": p["idx"],
                        "probability": response.answers[f"p{p['idx']}"].noul,
                        "model": response.model,
                    })
                f.flush()
                completed += 1
                if completed % 100 == 0 or completed == len(jobs):
                    print(f"  {completed}/{len(jobs)} requests", flush=True)

            await asyncio.gather(*(worker(row, chunk) for row, chunk in jobs))
