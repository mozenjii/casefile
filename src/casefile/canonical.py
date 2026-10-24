"""Canonical JSON: exactly one byte string for each value, so hashes over events are stable.

The event log chains events by hashing their serialised form. If two runs, two library versions or
two machines could serialise the same event differently, the chain would break for reasons that have
nothing to do with tampering. So serialisation here is deliberately narrow:

* keys sorted, no insignificant whitespace, UTF-8 with non-ASCII text kept as is;
* only ``None``, ``bool``, ``int``, ``str``, ``list`` and ``dict`` with ``str`` keys;
* no floats at all. Their text form differs between serialisers and they invite rounding bugs in
  money. Costs are integer micro-dollars and token counts are integers.

Anything else raises ``CanonicalJsonError`` instead of being coerced, because a silent coercion (a
tuple becoming a list, an int key becoming a string) would make two different values hash the same.
"""

from __future__ import annotations

import hashlib
import json
from typing import cast

# A value canonical JSON accepts. Floats are excluded on purpose; see the module docstring.
type JsonValue = bool | int | str | list[JsonValue] | dict[str, JsonValue] | None


class CanonicalJsonError(ValueError):
    """Raised for a value with no single canonical form: a float, a non-str key, another type."""


def canonical_json(value: object) -> bytes:
    """Serialise ``value`` to its canonical JSON bytes, or raise ``CanonicalJsonError``.

    The argument is typed ``object`` because the point of this function is to check values that
    arrive from outside the type checker's view (a model's output, a row read back from disk).
    """
    _check(value, path="$")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _check(value: object, path: str) -> None:
    """Walk ``value`` and raise at the first part canonical JSON refuses, naming where it is."""
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        raise CanonicalJsonError(f"{path}: float {value!r} is not allowed; use an integer unit")
    # isinstance cannot say what a list or dict holds; the walk below is what establishes it.
    if isinstance(value, list):
        for index, item in enumerate(cast(list[object], value)):
            _check(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"{path}: key {key!r} is not a string")
            _check(item, f"{path}.{key}")
        return
    raise CanonicalJsonError(f"{path}: {type(value).__name__} is not a JSON value")
