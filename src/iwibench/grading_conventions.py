"""Deterministic semantic and output-convention checks."""

from __future__ import annotations

import json
import re
import unicodedata


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _word(text: str) -> str:
    return _normal(text).rstrip(".!?")


def _result(content: bool, behaviour: bool) -> dict[str, bool]:
    return {"content": bool(content), "behaviour": bool(behaviour),
            "success": bool(content and behaviour)}


def output_code(item: dict, output: str) -> dict[str, bool]:
    """Correct ordinary labels preserve content but do not meet the new code."""
    content = _word(output) in {_word(item["class"]), _word(item["label"])}
    return _result(content, output.strip() == item["label"])


def classification_content(item: dict, output: str) -> dict[str, bool]:
    content = _word(output) in {_word(value) for value in item["answers"]}
    return _result(content, content)


def terminology(item: dict, output: str) -> dict[str, bool]:
    parts = output.strip().split(";")
    fields = item["fields"]
    content = len(parts) == len(fields) and all(
        _word(part) in {_word(value) for value in options}
        for part, options in zip(parts, fields)
    )
    preferred = item.get("preferred")
    behaviour = bool(output.strip()) and (preferred is None or _word(parts[0]) == _word(preferred))
    return _result(content, behaviour)


def verbatim(item: dict, output: str) -> dict[str, bool]:
    content = output.strip() in item["answers"]
    return _result(content, content)


def _json_object(output: str) -> dict | None:
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("Nonstandard JSON number")

    try:
        value = json.loads(output, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _prose_answers(values: dict[str, str], *, sentences_only: bool = False) -> set[str]:
    """Whole-answer forms; mentioning a value inside a false statement fails."""
    if set(values) == {"name"}:
        name = values["name"]
        sentences = [f"The person's name is {name}.", f"Their name is {name}.",
                     f"The name is {name}."]
        short = [name, f"Name: {name}", f"Person: {name}"]
    elif set(values) == {"city"}:
        city = values["city"]
        sentences = [f"The city is {city}.", f"They live in {city}.",
                     f"Their city is {city}.", f"The person's city is {city}."]
        short = [city, f"City: {city}", f"Home city: {city}"]
    elif set(values) == {"name", "city"}:
        name, city = values["name"], values["city"]
        sentences = [f"{name} lives in {city}.",
                     f"The person's name is {name} and they live in {city}.",
                     f"The person's name is {name} and their city is {city}.",
                     f"Their name is {name} and their city is {city}."]
        short = [f"{name}; {city}", f"{name}, {city}",
                 f"Name: {name}\nCity: {city}", f"Name: {name}; City: {city}",
                 f"Name: {name}, City: {city}"]
    else:
        raise ValueError("Unsupported extraction fields")
    return {_word(value) for value in sentences + ([] if sentences_only else short)}


def _prose_matches(item: dict, output: str, *, sentences_only: bool = False) -> bool:
    # Bold field names/values do not change their meaning. No arbitrary text is
    # discarded: negation, extra claims and reversed field assignments remain.
    plain = output.replace("**", "").replace("’", "'")
    return _word(plain) in _prose_answers(item["values"], sentences_only=sentences_only)


def _extraction_content(item: dict, output: str) -> bool:
    parsed = _json_object(output)
    values = item["values"]
    expected = {item["json_keys"][key]: value for key, value in values.items()}
    if parsed is not None:
        return parsed == expected or parsed == values
    return _prose_matches(item, output)


def structured_json(item: dict, output: str) -> dict[str, bool]:
    parsed = _json_object(output)
    expected = {item["json_keys"][key]: value for key, value in item["values"].items()}
    behaviour = parsed is not None and set(parsed) == set(expected) and all(
        isinstance(value, str) for value in parsed.values()
    )
    return _result(_extraction_content(item, output), behaviour)


def extraction_content(item: dict, output: str) -> dict[str, bool]:
    content = _extraction_content(item, output)
    return _result(content, content)


def prose_extraction(item: dict, output: str) -> dict[str, bool]:
    content = _prose_matches(item, output, sentences_only=True)
    behaviour = bool(output.strip()) and _json_object(output) is None and output.strip().endswith(".")
    return _result(content, behaviour)


CHECKERS = {
    "convention_output_code": output_code,
    "convention_classification_content": classification_content,
    "convention_terminology": terminology,
    "convention_verbatim": verbatim,
    "convention_structured_json": structured_json,
    "convention_extraction_content": extraction_content,
    "convention_prose_extraction": prose_extraction,
}
