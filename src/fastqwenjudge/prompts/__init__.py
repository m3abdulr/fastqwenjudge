"""System prompts and output schemas. The prompt texts live next to this file so they can be read and diffed
as plain text; judges load them unchanged."""
from importlib import resources


def load(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


def obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array_of(name: str, properties: dict) -> dict:
    return obj({name: {"type": "array", "items": obj(properties)}})


NULLABLE_INT = {"type": ["integer", "null"]}

VERDICTS = array_of("items", {"id": {"type": "integer"}, "evidence": {"type": "string"},
                              "satisfied": {"type": "boolean"}})

PLANS = array_of("plans", {
    "id": {"type": "integer"},
    "kind": {"type": "string", "enum": ["count", "absence", "universal", "other"]},
    "subject": {"type": "string"},
    "question": {"type": "string"},
    "comparator": {"type": "string", "enum": ["eq", "ge", "le", "none"]},
    "expected": NULLABLE_INT,
})

OBSERVATIONS = array_of("observations", {
    "id": {"type": "integer"}, "evidence": {"type": "string"}, "count": NULLABLE_INT, "total": NULLABLE_INT,
    "matching": NULLABLE_INT, "uncertain": {"type": "boolean"},
})


def numbered(lines) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
