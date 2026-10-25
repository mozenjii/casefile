"""Fuzz tests: random input against the parts of the system that take untrusted data.

Each property is one the code promises and an attacker would like broken: canonical JSON gives one
byte string per value, the grounding check never accepts words that are not a whole field, a model
reply can never crash the parser, and redaction never leaves an identifier behind.
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from casefile.agents import SourceDocument, ground
from casefile.canonical import JsonValue, canonical_json
from casefile.generator import DEFECTS, generate
from casefile.identifiers import find_identifiers, redact
from casefile.llm import parse_json
from casefile.quotes import fields
from casefile.rules import FACT_TYPES

json_values: st.SearchStrategy[JsonValue] = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=20,
)


@given(json_values)
def test_canonical_json_round_trips_and_ignores_key_order(value: JsonValue) -> None:
    encoded = canonical_json(value)
    assert json.loads(encoded) == value
    if isinstance(value, dict):
        reordered = dict(reversed(list(value.items())))
        assert canonical_json(reordered) == encoded


CASES = [generate(seed, defect) for seed in (1, 2) for defect in DEFECTS]
DOCUMENTS = [SourceDocument(d.id, d.title, d.text) for case in CASES for d in case.documents]


@settings(max_examples=400)
@given(
    st.sampled_from(DOCUMENTS),
    st.sampled_from(sorted(FACT_TYPES)),
    st.integers(min_value=0, max_value=400),
    st.integers(min_value=1, max_value=40),
    st.one_of(st.booleans(), st.integers(0, 3000), st.text(max_size=12)),
)
def test_grounding_never_accepts_words_that_are_not_a_whole_field(
    document: SourceDocument, fact: str, start: int, width: int, value: str | int | bool
) -> None:
    observation: JsonValue = {"fact": fact, "value": value, "start": start, "end": start + width}
    kept, _ = ground(document, {"observations": [observation]})
    allowed = fields(fact, document.text)
    for item in kept:
        assert isinstance(item, dict)
        assert (item["start"], item["end"]) in allowed


@settings(max_examples=300)
@given(st.text(max_size=300))
def test_no_model_reply_can_crash_the_parser(reply: str) -> None:
    parsed = parse_json(reply)
    assert parsed is None or isinstance(parsed, dict)


@given(st.text(alphabet=" abcdefxyz.,\n", max_size=40), st.sampled_from(["ssn", "phone", "email"]))
def test_redaction_never_leaves_an_identifier_behind(padding: str, kind: str) -> None:
    planted = {
        "ssn": "-".join(("123", "45", "6789")),
        "phone": "-".join(("212", "867", "5309")),
        "email": "jane" + "@" + "clinic.org",
    }[kind]
    text = f"{padding} {planted} {padding}"
    assert find_identifiers(redact(text)) == []
    assert planted not in redact(text)
