"""Run jev's noul primitive over dataset rows to predict founder success."""

import asyncio

from jev import AsyncJevClient

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
