"""Language-model providers behind the one interface the specialists use.

The specialists call a model as ``Call``: a JSON request in, a JSON response out. This module turns
a provider into that shape. Which provider is chosen by environment variables, so a fresh clone
works with no key at all and switching to a real model needs no code change:

=====================  ======================================================================
``CASEFILE_PROVIDER``  a preset below; default ``reference`` (the deterministic stand-in)
``CASEFILE_MODEL``     the model id, overriding the preset's default
``CASEFILE_BASE_URL``  the endpoint, overriding the preset's (any OpenAI-compatible server)
``CASEFILE_API_KEY``   the key, overriding the preset's own variable (e.g. ``OPENROUTER_API_KEY``)
``CASEFILE_EFFORT``    Anthropic only: ``low`` to ``max``; the model's default when unset
``CASEFILE_PRICE_INPUT`` / ``CASEFILE_PRICE_OUTPUT``  USD per million tokens, for cost figures
=====================  ======================================================================

Two backends cover the field. ``anthropic`` uses the official Anthropic SDK with structured JSON
output and the server-side refusal fallback. ``openai`` uses the official OpenAI SDK against any
OpenAI-compatible endpoint, which is how OpenRouter, Together, Groq, Fireworks, DeepInfra and a
local Ollama all serve open-weights models. Both SDKs retry rate limits and server errors
themselves; a successful reply is then recorded in the case's log, so a retry never repeats work
that already happened.

Every reply carries ``usage``: model, input and output tokens, and the cost in micro-dollars when a
price is known. A cost is never reported as zero when it is simply unknown.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from casefile.canonical import JsonValue, canonical_json
from casefile.prompts import Prompt, extract_prompt, write_prompt
from casefile.reference_model import reference_model

if TYPE_CHECKING:
    import httpx2
    from anthropic.types.beta import BetaOutputConfigParam

type Call = Callable[[JsonValue], JsonValue]
type Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ConfigError(ValueError):
    """The provider settings are incomplete or contradictory; the message says what to set."""


class ModelRefused(RuntimeError):
    """The model declined to answer (after any fallback the provider offers)."""


@dataclass(frozen=True)
class Preset:
    """A named provider: which backend, where, which key variable, and a default model."""

    backend: str
    base_url: str | None = None
    key_variable: str | None = None
    model: str | None = None
    price_input: float | None = None
    price_output: float | None = None


PRESETS: Mapping[str, Preset] = {
    "reference": Preset("reference", model="reference"),
    # Recommended for most users: one key reaches many hosts. Default model and its price were
    # read from OpenRouter's public model list on 2026-10-25; override both if they have moved.
    "openrouter": Preset(
        "openai",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "deepseek/deepseek-v4-pro",
        0.209,
        0.418,
    ),
    # Prices are Anthropic's published first-party rates as of 2026-10-06.
    "anthropic": Preset("anthropic", None, "ANTHROPIC_API_KEY", "claude-opus-5-5", 4.0, 20.0),
    # These hosts serve many models; there is no safe default, so CASEFILE_MODEL is required.
    "together": Preset("openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": Preset("openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "fireworks": Preset("openai", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "deepinfra": Preset("openai", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "ollama": Preset("openai", "http://localhost:11434/v1", None),
    "openai-compatible": Preset("openai", None, None),
}
"""Every provider name ``CASEFILE_PROVIDER`` accepts."""

ANTHROPIC_PRICES: Mapping[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
"""USD per million input/output tokens, Anthropic first-party rates as of 2026-10-06."""

EFFORTS: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")
"""The effort levels Claude accepts."""

_FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5"})
"""Models that accept ``fallbacks: "default"`` on the Claude API."""


@dataclass(frozen=True)
class Settings:
    """A resolved provider configuration."""

    provider: str
    backend: str
    model: str
    base_url: str | None
    api_key: str | None = field(repr=False)
    effort: str | None
    price_input: float | None
    price_output: float | None
    max_tokens: int

    def cost(self, input_tokens: int, output_tokens: int) -> int | None:
        """Cost of one call in micro-dollars (millionths of a dollar), or None with no price.

        Integers, because the event log refuses floats: two serialisers disagree about them, and
        a hash chain cannot. A price per million tokens times tokens is exactly micro-dollars.
        """
        if self.price_input is None or self.price_output is None:
            return None
        return round(input_tokens * self.price_input + output_tokens * self.price_output)


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    """Resolve the provider from environment variables (``os.environ`` by default)."""
    values = os.environ if env is None else env
    name = values.get("CASEFILE_PROVIDER", "reference").strip().lower() or "reference"
    preset = PRESETS.get(name)
    if preset is None:
        raise ConfigError(f"CASEFILE_PROVIDER={name!r} is not one of: {', '.join(PRESETS)}")

    model = values.get("CASEFILE_MODEL") or preset.model
    if model is None:
        raise ConfigError(f"provider {name!r} serves many models: set CASEFILE_MODEL")
    base_url = values.get("CASEFILE_BASE_URL") or preset.base_url
    if preset.backend == "openai" and base_url is None:
        raise ConfigError(f"provider {name!r} needs an endpoint: set CASEFILE_BASE_URL")
    if base_url is not None:
        _check_endpoint(base_url)

    api_key = values.get("CASEFILE_API_KEY") or (
        values.get(preset.key_variable) if preset.key_variable else None
    )
    if preset.backend != "reference" and not api_key:
        if name == "ollama":
            api_key = "ollama"  # a local server ignores the key, but the client wants one
        else:
            wanted = preset.key_variable or "CASEFILE_API_KEY"
            raise ConfigError(f"provider {name!r} needs an API key: set {wanted}")

    default_price = (preset.price_input, preset.price_output) if model == preset.model else None
    if preset.backend == "anthropic":
        default_price = ANTHROPIC_PRICES.get(model, default_price)
    price_input = _price(values, "CASEFILE_PRICE_INPUT", default_price, 0)
    price_output = _price(values, "CASEFILE_PRICE_OUTPUT", default_price, 1)
    effort = values.get("CASEFILE_EFFORT") or None
    if effort is not None and effort not in EFFORTS:
        raise ConfigError(f"CASEFILE_EFFORT={effort!r} is not one of: {', '.join(EFFORTS)}")
    max_tokens = 16000 if preset.backend == "anthropic" else 4096
    return Settings(
        name,
        preset.backend,
        model,
        base_url,
        api_key,
        effort,
        price_input,
        price_output,
        max_tokens,
    )


def _price(
    values: Mapping[str, str],
    variable: str,
    default: tuple[float | None, float | None] | None,
    index: int,
) -> float | None:
    """Read a price override, falling back to the preset's price for its own default model."""
    raw = values.get(variable)
    if raw:
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(f"{variable}={raw!r} is not a number of dollars") from None
    return default[index] if default else None


@dataclass(frozen=True)
class Completion:
    """What a backend returns: the reply text and what it cost in tokens."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class Backend(Protocol):
    """One provider's way of answering a system prompt and a user message with schema-bound JSON."""

    def complete(self, system: str, user: str, schema: dict[str, object]) -> Completion:
        """Return the model's reply to one prompt."""
        ...


class AnthropicBackend:
    """Claude through the official Anthropic SDK: structured JSON output, refusal fallback."""

    def __init__(self, settings: Settings, http_client: httpx2.Client | None = None) -> None:
        """Build the SDK client; ``http_client`` lets tests replace the network."""
        try:
            import anthropic
        except ImportError as missing:
            raise ConfigError("install the extra: uv sync --extra anthropic") from missing
        self._sdk = anthropic
        self._settings = settings
        self._client = anthropic.Anthropic(
            api_key=settings.api_key, max_retries=3, timeout=120.0, http_client=http_client
        )

    def complete(self, system: str, user: str, schema: dict[str, object]) -> Completion:
        """Ask Claude for JSON matching ``schema``; raise ``ModelRefused`` on a final refusal."""
        settings, omit = self._settings, self._sdk.omit
        output_config: BetaOutputConfigParam = {"format": {"type": "json_schema", "schema": schema}}
        if settings.effort:
            output_config["effort"] = cast(Effort, settings.effort)  # validated in settings
        with_fallback = settings.model in _FALLBACK_MODELS
        response = self._client.beta.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            fallbacks="default" if with_fallback else omit,
            betas=["server-side-fallback-2026-07-01"] if with_fallback else omit,
        )
        if response.stop_reason == "refusal":
            raise ModelRefused(f"{settings.model} declined to answer")
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return Completion(text, usage.input_tokens, usage.output_tokens, response.model)


class OpenAICompatibleBackend:
    """Any OpenAI-compatible chat endpoint through the official OpenAI SDK."""

    def __init__(self, settings: Settings, http_client: httpx2.Client | None = None) -> None:
        """Build the SDK client; ``http_client`` lets tests replace the network."""
        try:
            import openai
        except ImportError as missing:
            raise ConfigError("install the extra: uv sync --extra openai") from missing
        self._settings = settings
        self._client = openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=3,
            timeout=120.0,
            http_client=http_client,
        )

    def complete(self, system: str, user: str, schema: dict[str, object]) -> Completion:
        """Ask for a JSON object; the schema is stated in the prompt, which every host honours."""
        settings = self._settings
        instructions = f"{system}\n\nThe JSON must match this schema:\n{json.dumps(schema)}"
        response = self._client.chat.completions.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice else ""
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        return Completion(text, tokens_in, tokens_out, response.model or settings.model)


class LanguageModel:
    """Answer the specialists' requests with a real model, in the reference model's JSON shape."""

    def __init__(self, backend: Backend, settings: Settings) -> None:
        """Wrap a backend configured by ``settings``."""
        self._backend = backend
        self._settings = settings

    def __call__(self, request: JsonValue) -> JsonValue:
        """Build the prompt for the task, ask the model, and parse its JSON reply."""
        if not isinstance(request, dict):
            raise ValueError("a model request is a JSON object")
        task = request.get("task")
        prompts: dict[object, Callable[[dict[str, JsonValue]], Prompt]] = {
            "extract": extract_prompt,
            "write": write_prompt,
        }
        if task not in prompts:
            raise ValueError(f"no prompt for task {task!r}")
        system, user, schema = prompts[task](request)
        completion = self._backend.complete(system, user, schema)
        usage: dict[str, JsonValue] = {
            "model": completion.model,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "cost_microusd": self._settings.cost(completion.input_tokens, completion.output_tokens),
        }
        parsed = parse_json(completion.text)
        if not isinstance(parsed, dict):
            # Left for the grounding check to reject as malformed, with the usage still recorded.
            return {"unparsed": completion.text[:500], "usage": usage}
        return {**parsed, "usage": usage}


MAX_REPLY_CHARS = 200_000
"""Replies longer than this are refused unread: nothing the specialists ask for is that long."""


def parse_json(text: str) -> JsonValue:
    """Parse a model's JSON reply, or return None if it is not a recordable JSON object.

    Whatever surrounds the outermost braces (a Markdown fence, a polite sentence) is ignored by
    slicing, not by a regular expression, so no reply can make parsing slow. A reply that parses
    but could not be recorded (a float, a number too long to convert, nesting deep enough to blow
    the stack) is refused here, so it is rejected and recorded instead of crashing the step and
    being paid for again on every resume.
    """
    if len(text) > MAX_REPLY_CHARS:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        parsed: JsonValue = json.loads(text[start : end + 1])
        canonical_json(parsed)
    except (ValueError, RecursionError):  # JSONDecodeError and CanonicalJsonError are ValueErrors
        return None
    return parsed


def build_model(settings: Settings | None = None, http_client: httpx2.Client | None = None) -> Call:
    """Return the ``Call`` the specialists should use for ``settings`` (from the environment)."""
    resolved = settings or settings_from_env()
    if resolved.backend == "reference":
        return reference_model
    if resolved.backend == "anthropic":
        return LanguageModel(AnthropicBackend(resolved, http_client), resolved)
    return LanguageModel(OpenAICompatibleBackend(resolved, http_client), resolved)


def describe(settings: Settings) -> str:
    """One line naming the provider and model, never the key or credentials in the endpoint."""
    where = f" at {_without_credentials(settings.base_url)}" if settings.base_url else ""
    return f"{settings.provider} ({settings.model}{where})"


def _without_credentials(url: str) -> str:
    """Drop any ``user:password@`` from a URL before it is printed."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_endpoint(url: str) -> None:
    """Refuse an endpoint that would send the key in the clear to another machine."""
    parts = urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOOPBACK):
        return
    shown = _without_credentials(url)
    raise ConfigError(f"endpoint {shown!r} must use https (plain http only for this machine)")


__all__ = [
    "PRESETS",
    "AnthropicBackend",
    "ConfigError",
    "LanguageModel",
    "ModelRefused",
    "OpenAICompatibleBackend",
    "Settings",
    "build_model",
    "describe",
    "parse_json",
    "read_env_file",
    "settings_for_repository",
    "settings_from_env",
]


def read_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file; a missing file reads as empty.

    Blank lines and ``#`` comments are skipped and matching quotes around a value are removed.
    Nothing is exported: callers merge the result *under* the real environment, so a variable
    set in the shell always wins over the file.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.strip():
            values[key.strip()] = value
    return values


def settings_for_repository(root: Path) -> Settings:
    """Resolve settings from ``root/.env`` overlaid by the real environment."""
    return settings_from_env({**read_env_file(root / ".env"), **os.environ})
