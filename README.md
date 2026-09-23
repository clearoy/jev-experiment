# jev_experiment

## Setup

```bash
pip install -r requirements.txt
```

Put your TypeSafe API key in `.env`:

```
TYPESAFE_API_KEY=<your_key>
```

## Calling jev

```python
from jev import JevClient

with JevClient() as client:
    client.noul(state, instructions="Is this about billing?")
    client.choice(state, instructions="What is the tone?", criteria={"calm": None, "angry": None})
    client.score(state, instructions="How urgent is this?", criteria=["can wait", "this week", "today"])
    client.ask(state, questions={"tone": Choice(...), "urgency": Score(...)})
    client.list_models()
```

Async equivalent: `from jev import AsyncJevClient`, same methods, `await`-ed.


## Test run

```bash
python script/test_jev.py
```

## Experiments

Put `vcbench_final_public.csv`, `vcbench_final_private.csv` and `policies/policies.csv` under `data/`. Outputs go to `result/` (both folders are local-only).

| Script | What it does |
|---|---|
| `script/jev_vanilla.py` | One yes/no question per private founder, threshold 0.5 |
| `script/jev_vanilla_fit_threshold.py` | Same question on the public set, fits the F0.5 threshold, re-scores the private run |
| `script/jev_with_policies.py [--policy-limit N]` | One question per founder × policy, L1 logistic regression fit on public, evaluated on private |

## Results (private set, n = 4,500)

| Method | Precision | Recall | AUC | F0.5 |
|---|---|---|---|---|
| Jev + 201 policies | 42.4 | 24.0 | 74.2 | **36.7** |
| Jev + 30 policies | 34.8 | 26.9 | 71.5 | 32.9 |
| Jev vanilla, fitted threshold | 28.3 | 10.1 | 65.1 | 20.8 |
| Jev vanilla, threshold 0.5 | 14.1 | 52.1 | 65.1 | 16.5 |

Open `dashboard.html` in a browser for the same scores as a dashboard.
