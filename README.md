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
