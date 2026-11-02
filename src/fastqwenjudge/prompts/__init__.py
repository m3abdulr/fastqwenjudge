"""System prompts and output schemas. The prompt texts live next to this file so they can be read and diffed
as plain text; judges load them unchanged."""
from importlib import resources


def load(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


def obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array_of(name: str, properties: dict) -> dict:
    return obj({name: {"type": "array", "items": obj(properties)}})


# No maxLength on the evidence: with any bound, xgrammar 0.2.8 builds each token's mask ~10x slower (224 against 21 us),
# which left the GPU waiting on it (docs/serving.md). The request's max_tokens bounds an answer that loops.
VERDICTS = array_of("items", {"id": {"type": "integer"}, "evidence": {"type": "string"},
                              "satisfied": {"type": "boolean"}})

CHECKLIST = obj({"requirements": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20}})


def numbered(lines) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
