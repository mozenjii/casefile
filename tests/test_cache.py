"""Tests for the model response cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from casefile.cache import ResponseCache, cached
from casefile.canonical import JsonValue
from casefile.llm import ModelRefused


class Counting:
    """A fake model that counts how often it is really asked."""

    def __init__(self) -> None:
        """Start at zero calls."""
        self.calls = 0

    def __call__(self, request: JsonValue) -> JsonValue:
        """Answer with the request and the call number."""
        self.calls += 1
        return {"echo": request, "n": self.calls}


def test_an_identical_request_is_answered_from_the_cache(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    model = Counting()
    ask = cached(model, cache, "openrouter/deepseek")
    first = ask({"task": "extract", "text": "a"})
    assert ask({"text": "a", "task": "extract"}) == first
    assert model.calls == 1
    assert cache.size() == 1
    cache.close()


def test_the_cache_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    first = ResponseCache(path)
    cached(Counting(), first, "m")({"q": 1})
    first.close()
    model = Counting()
    second = ResponseCache(path)
    assert cached(model, second, "m")({"q": 1}) == {"echo": {"q": 1}, "n": 1}
    assert model.calls == 0
    second.close()


def test_a_different_model_or_request_asks_again(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    model = Counting()
    cached(model, cache, "model-a")({"q": 1})
    cached(model, cache, "model-b")({"q": 1})
    cached(model, cache, "model-a")({"q": 2})
    assert model.calls == 3
    cache.close()


def test_a_null_answer_is_cached_too(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    calls: list[JsonValue] = []

    def silent(request: JsonValue) -> JsonValue:
        calls.append(request)
        return None

    ask = cached(silent, cache, "m")
    assert ask({"q": 1}) is None
    assert ask({"q": 1}) is None
    assert len(calls) == 1
    cache.close()


def test_a_refusal_is_not_cached(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")

    def refusing(request: JsonValue) -> JsonValue:
        raise ModelRefused("declined")

    with pytest.raises(ModelRefused):
        cached(refusing, cache, "m")({"q": 1})
    assert cache.size() == 0
    cache.close()
