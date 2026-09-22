"""Thin convenience wrapper around the ``typesafe_sdk`` client for Jev (System One).

Wraps ``TypeSafeClient`` / ``AsyncTypeSafeClient`` and exposes one method per
question primitive (choice / score / noul) in addition to the raw multi-question
``ask`` / ``aask`` calls, so callers don't need to unwrap ``SystemOneResponse``
by hand for the common single-question case.
"""

from __future__ import annotations

from typing import Any, Mapping

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Question,
    Score,
    TypeSafeClient,
)

JSONContent = str | Mapping[str, Any] | list[Any]


class JevClient:
    """Synchronous convenience client for Jev / System One."""

    def __init__(self, **kwargs: Any) -> None:
        self._client = TypeSafeClient(**kwargs)

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def ask(self, state: JSONContent, questions: Mapping[str, Question], **kwargs: Any):
        """Answer one or more named questions about ``state``. Returns SystemOneResponse."""
        return self._client.system_one(state=state, questions=questions, **kwargs)

    def choice(self, state: JSONContent, instructions: str, criteria: Mapping[str, str | None], **kwargs: Any):
        """Ask a single Choice question and return the answer directly."""
        response = self.ask(state, {"q": Choice(instructions=instructions, criteria=criteria)}, **kwargs)
        return response.answers["q"]

    def score(self, state: JSONContent, instructions: str, criteria: list[str], **kwargs: Any):
        """Ask a single Score question and return the answer directly."""
        response = self.ask(state, {"q": Score(instructions=instructions, criteria=criteria)}, **kwargs)
        return response.answers["q"]

    def noul(self, state: JSONContent, instructions: str, criteria: Mapping[str, str] | None = None, **kwargs: Any):
        """Ask a single Noul (yes/no) question and return the probability that it is yes."""
        response = self.ask(state, {"q": Noul(instructions=instructions, criteria=criteria)}, **kwargs)
        return response.answers["q"].noul

    def list_models(self, **kwargs: Any):
        return self._client.models.list(**kwargs)


class AsyncJevClient:
    """Asynchronous convenience client for Jev / System One."""

    def __init__(self, **kwargs: Any) -> None:
        self._client = AsyncTypeSafeClient(**kwargs)

    async def __aenter__(self) -> "AsyncJevClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def ask(self, state: JSONContent, questions: Mapping[str, Question], **kwargs: Any):
        return await self._client.system_one(state=state, questions=questions, **kwargs)

    async def choice(self, state: JSONContent, instructions: str, criteria: Mapping[str, str | None], **kwargs: Any):
        response = await self.ask(state, {"q": Choice(instructions=instructions, criteria=criteria)}, **kwargs)
        return response.answers["q"]

    async def score(self, state: JSONContent, instructions: str, criteria: list[str], **kwargs: Any):
        response = await self.ask(state, {"q": Score(instructions=instructions, criteria=criteria)}, **kwargs)
        return response.answers["q"]

    async def noul(self, state: JSONContent, instructions: str, criteria: Mapping[str, str] | None = None, **kwargs: Any):
        response = await self.ask(state, {"q": Noul(instructions=instructions, criteria=criteria)}, **kwargs)
        return response.answers["q"].noul

    async def list_models(self, **kwargs: Any):
        return await self._client.models.list(**kwargs)
