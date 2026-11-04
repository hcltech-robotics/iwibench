"""Deterministic development episodes; private items stay in the trusted service."""

from __future__ import annotations

import hashlib
import json
import random
from importlib.resources import files
from pathlib import Path
from typing import Any

from .config import MODELS

FAMILY_VERSION = "0.2.2"
MODEL_IDS = ("google/gemma-3-1b-it", "google/gemma-3-4b-it")
AVAILABLE_FAMILIES = (
    "alias_acquisition", "output_code_remapping", "lexical_inhibition",
    "terminology_preference", "default_response_language", "structured_response_default",
    "answer_length_control", "factual_update", "selective_factual_withholding",
    "resistance_to_false_suggestions",
)
FAMILIES = AVAILABLE_FAMILIES
PLANNED_FAMILIES = ()
PRIVATE_GROUPS = ("calibration", "targets", "nearby_controls", "general_controls", "override_controls")
REFERENCE_DATE = "2026-03-01"

# Each property is one semantic case. Template changes do not increase this count.
CONCEPTS = {
    "development": [
        ("triangle", [("straight sides", 3), ("vertices", 3), ("dimensions", 2)]),
        ("square", [("straight sides", 4), ("vertices", 4), ("right angles", 4)]),
        ("circle", [("vertices", 0), ("straight sides", 0), ("dimensions", 2)]),
        ("cube", [("flat faces", 6), ("vertices", 8), ("edges", 12)]),
        ("sphere", [("flat faces", 0), ("edges", 0), ("dimensions", 3)]),
        ("pentagon", [("straight sides", 5), ("vertices", 5), ("diagonals", 5)]),
        ("hexagon", [("straight sides", 6), ("vertices", 6), ("diagonals", 9)]),
        ("octagon", [("straight sides", 8), ("vertices", 8), ("diagonals", 20)]),
        ("cylinder", [("flat faces", 2), ("curved surfaces", 1), ("vertices", 0)]),
        ("cone", [("flat faces", 1), ("curved surfaces", 1), ("vertices", 1)]),
    ],
    "test": [
        ("rectangle", [("straight sides", 4), ("vertices", 4), ("right angles", 4)]),
        ("heptagon", [("straight sides", 7), ("vertices", 7), ("diagonals", 14)]),
        ("decagon", [("straight sides", 10), ("vertices", 10), ("diagonals", 35)]),
        ("square pyramid", [("flat faces", 5), ("vertices", 5), ("edges", 8)]),
        ("triangular prism", [("flat faces", 5), ("vertices", 6), ("edges", 9)]),
    ],
}
ALIASES = {
    "development": ["quince", "sprocket", "truffle", "saffron", "fjord", "quartz", "goblet", "cobalt", "tulip", "tundra", "badger", "larch"],
    "test": ["gannet", "dormer", "wombat", "cairn", "tussock", "plover"],
}

# Lexical answers contain two independently checked fields. The target prompt
# specifies their order, but never mentions the inhibited words or paraphrases.
# tuple: conventional word, banned forms, question, accepted first field, second field.
LEXICAL_POOLS = {
    "development": [
        [("cat", ["cat", "cats", "kitten", "kittens"], "Which common domestic animal purrs, and is it a mammal or a reptile?", ["feline", "a feline", "domestic feline", "domesticated feline"], ["mammal", "a mammal"]),
         ("dog", ["dog", "dogs", "puppy", "puppies"], "Which common domestic animal barks, and is it a mammal or a reptile?", ["canine", "a canine", "domestic canine", "domesticated canine"], ["mammal", "a mammal"])],
        [("doctor", ["doctor", "doctors"], "Name the professional who diagnoses and treats patients, and name their industry.", ["physician", "a physician", "medical practitioner"], ["healthcare", "health care", "medicine"]),
         ("teacher", ["teacher", "teachers"], "Name the professional who teaches pupils in a school, and name their industry.", ["educator", "an educator", "school educator", "instructor"], ["education"])],
        [("bicycle", ["bicycle", "bicycles"], "Name the usual two-wheeled pedal-powered road vehicle, and state its number of wheels.", ["bike", "a bike", "pedal cycle", "a pedal cycle", "cycle"], ["2", "two"]),
         ("car", ["car", "cars"], "Name the usual four-wheeled passenger road vehicle, and state its usual number of wheels.", ["automobile", "an automobile", "motor vehicle"], ["4", "four"])],
        [("rain", ["rain", "rains", "raining", "rainfall"], "Name precipitation that falls from clouds as liquid drops, and give its physical state.", ["liquid precipitation", "falling water", "water droplets", "liquid water precipitation"], ["liquid"]),
         ("snow", ["snow", "snows", "snowing", "snowfall"], "Name precipitation made of ice crystals, and give its physical state.", ["frozen precipitation", "ice crystals", "frozen water crystals"], ["solid"])],
        [("water", ["water", "waters"], "Name the compound with formula H2O, and give the two elements it contains.", ["H2O", "dihydrogen monoxide"], ["hydrogen and oxygen", "oxygen and hydrogen", "hydrogen, oxygen", "H and O", "H, O"]),
         ("oxygen", ["oxygen"], "Name the element with atomic number 8, and give its chemical symbol.", ["the element with atomic number eight", "element 8", "O"], ["O"])],
        [("sun", ["sun", "suns", "sunlight"], "Name the star at the centre of our solar system, and classify it as a star or a planet.", ["Sol", "our solar system's central star", "our central star", "the solar system's star"], ["star", "a star"]),
         ("moon", ["moon", "moons", "moonlight"], "Name Earth's principal natural satellite, and give the planet it orbits.", ["Luna", "Earth's natural satellite", "our natural satellite"], ["Earth"])],
        [("heart", ["heart", "hearts"], "Name the muscular organ that pumps blood around the body, and name the fluid it pumps.", ["cardiac organ", "the cardiac organ", "cardiac muscle", "cardiac pump"], ["blood"]),
         ("lungs", ["lung", "lungs"], "Name the paired respiratory organs in the chest, and give their main function.", ["pulmonary organs", "respiratory organs", "paired respiratory organs"], ["gas exchange", "exchange gases", "breathing"])],
        [("ocean", ["ocean", "oceans"], "Name a vast body of salt water covering much of Earth, and say whether its water is salty or fresh.", ["sea", "a sea", "large sea", "vast sea"], ["salty", "salt water", "saltwater"]),
         ("river", ["river", "rivers"], "Give the general term for a large natural flowing watercourse, rather than the name of a particular example, and say whether it is flowing or still.", ["watercourse", "a watercourse", "flowing watercourse", "large stream"], ["flowing"])],
        [("tree", ["tree", "trees"], "Give the general term for a tall woody perennial plant with a trunk, rather than a species name, and name the process by which its green leaves make food.", ["woody perennial", "a woody perennial", "woody plant", "tall woody plant"], ["photosynthesis"]),
         ("flower", ["flower", "flowers", "flowering"], "Name the reproductive blossom of a plant, and name the transfer of pollen to its stigma.", ["blossom", "a blossom", "bloom"], ["pollination", "pollen transfer"])],
        [("piano", ["piano", "pianos"], "Name the keyboard instrument whose hammers strike strings, and name what its hammers strike.", ["pianoforte", "a pianoforte", "hammered-string keyboard instrument", "keyboard instrument"], ["strings"]),
         ("violin", ["violin", "violins"], "Name the small four-string bowed instrument also called a fiddle, and give its usual number of strings.", ["fiddle", "a fiddle"], ["4", "four"])],
    ],
    "test": [
        [("rabbit", ["rabbit", "rabbits"], "Name the long-eared small mammal that hops, and say whether it is a mammal or a reptile.", ["bunny", "a bunny", "lagomorph", "long-eared lagomorph"], ["mammal", "a mammal"]),
         ("horse", ["horse", "horses"], "Name the large domesticated equine used for riding, and say whether it is a mammal or a reptile.", ["equine", "an equine", "domesticated equine"], ["mammal", "a mammal"])],
        [("clock", ["clock", "clocks"], "Name a device that shows the time of day, and name what it measures.", ["timepiece", "a timepiece"], ["time"]),
         ("calendar", ["calendar", "calendars"], "Name a chart that organises days into months, and give the number of months in a year.", ["date chart", "almanac", "an almanac"], ["12", "twelve"])],
    ],
}

# Count nouns have equivalent singular and plural category answers. These are
# content alternatives only; the independent lexical checker still rejects
# inhibited singulars and plurals. Uncountable substances and proper names do
# not receive invented plurals.
NOUN_PLURALS = {
    "cat": "cats", "dog": "dogs", "doctor": "doctors", "teacher": "teachers",
    "bicycle": "bicycles", "car": "cars", "heart": "hearts", "ocean": "oceans",
    "river": "rivers", "tree": "trees", "flower": "flowers", "piano": "pianos",
    "violin": "violins", "rabbit": "rabbits", "horse": "horses", "clock": "clocks",
    "calendar": "calendars", "feline": "felines", "canine": "canines",
    "physician": "physicians", "educator": "educators", "instructor": "instructors",
    "bike": "bikes", "cycle": "cycles", "automobile": "automobiles", "sea": "seas",
    "watercourse": "watercourses", "blossom": "blossoms", "bloom": "blooms",
    "pianoforte": "pianofortes", "fiddle": "fiddles", "bunny": "bunnies",
    "lagomorph": "lagomorphs", "equine": "equines", "timepiece": "timepieces",
    "almanac": "almanacs",
}


def _lexical_names(word: str, alternatives: list[str]) -> list[str]:
    names = [word, *alternatives]
    for name in list(names):
        if name in NOUN_PLURALS:
            names.append(NOUN_PLURALS[name])
            names.append(("an " if name[0].lower() in "aeiou" else "a ") + name)
    names.extend("the " + name for name in list(names) if not name.lower().startswith(("a ", "an ", "the ")))
    return list(dict.fromkeys(names))

GENERAL_FACTS = [
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Portugal?", "Lisbon"), ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Canada?", "Ottawa"), ("What is the capital of Australia?", "Canberra"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for iron?", "Fe"), ("What is the chemical symbol for helium?", "He"),
    ("How many days are in a normal non-leap year?", "365"), ("How many minutes are in an hour?", "60"),
    ("How many seconds are in a minute?", "60"), ("How many centimetres are in a metre?", "100"),
]


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).digest()[:8], "big")


def _item(identifier: str, prompt: str, answers: list[str], **extra: Any) -> dict[str, Any]:
    return {"id": identifier, "prompt": prompt, "rule": "exact", "answers": answers, **extra}


def _facts() -> list[dict[str, Any]]:
    # Source checkout and wheel both use the same JSON bytes. The package builder
    # includes data/facts.json as iwibench/data/facts.json.
    installed = files("iwibench").joinpath("data/facts.json")
    if installed.is_file():
        return json.loads(installed.read_text(encoding="utf-8"))
    source = Path(__file__).resolve().parents[2] / "data" / "facts.json"
    return json.loads(source.read_text(encoding="utf-8"))


def _general_controls(seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs = rng.sample([(a, b) for a in range(2, 50) for b in range(2, 50)], 32)
    arithmetic = [_item(f"general:arithmetic:{i}", f"What is {a} + {b}? Give only the integer.", [str(a + b)]) for i, (a, b) in enumerate(pairs)]
    facts = [_item(f"general:fact:{i}", f"{q} Give only the answer.", [a]) for i, (q, a) in enumerate(GENERAL_FACTS)]
    strings = [f"{split}_{rng.randrange(10**9):09d}" for _ in range(16)]
    copies = [_item(f"general:copy:{i}", f"Copy the following string exactly, without quotation marks: {s}", [s], case_sensitive=True) for i, s in enumerate(strings)]
    return arithmetic + facts + copies


def _alias(ep: dict[str, Any], seed: int, split: str) -> None:
    pool = CONCEPTS[split]
    referent, properties = pool[seed % len(pool)]
    alias = ALIASES[split][seed % len(ALIASES[split])]
    ep["objective"] = f"Make the existing word '{alias}' refer to {referent}. Answer questions using that meaning without a definition in the target prompt. Preserve the ordinary meaning of '{referent}', unrelated concepts and explicit copying."
    ep["parameters"] = {"bindings": [{"alias": alias, "referent": referent}], "alias": alias, "referent": referent, "requires_single_token_alias": True}
    ep["calibration_requirements"] = {"ordinary_referent_accuracy": 1.0, "target_success_max_exclusive": 1.0, "native_tokenisation_required": True}
    for i, (property_name, number) in enumerate(properties):
        if split == "development":
            target_prompt = f"A geometry exercise mentions a {alias}. How many {property_name} does it have? Give only the integer."
        else:
            target_prompt = f"For a {alias}, write the count of {property_name}. Use a single integer."
        ep["targets"].append(_item(f"target:{i}", target_prompt, [str(number)]))
        ep["calibration"].append(_item(f"calibration:{i}", f"How many {property_name} does a {referent} have? Give only the integer.", [str(number)], target_id=f"target:{i}"))
        ep["nearby_controls"].append(_item(f"nearby:{i}", target_prompt.replace(alias, referent), [str(number)]))
    prop, number = properties[0]
    ep["public_practice"] = [_item("practice:0", f"Count the {prop} of a {alias}. Return only the integer.", [str(number)])]
    neighbour, neighbour_props = pool[(seed + 1) % len(pool)]
    for i, (prop, value) in enumerate(neighbour_props):
        ep["nearby_controls"].append(_item(f"nearby:other:{i}", f"How many {prop} does a {neighbour} have? Give only the integer.", [str(value)]))
    ep["override_controls"] = [_item("override:copy", f"Copy this word exactly without quotation marks: {alias}", [alias], case_sensitive=True)]
    ep["case_counts"] = {"target_semantic_cases": len(properties), "parameter_pool_size": len(pool)}


def _lexical(ep: dict[str, Any], seed: int, split: str) -> None:
    pool = LEXICAL_POOLS[split]
    selected = pool[seed % len(pool)]
    banned = [form for entry in selected for form in entry[1]]
    words = [entry[0] for entry in selected]
    ep["objective"] = f"Avoid these words and their listed forms in generated answers: {', '.join(banned)}. Preserve every requested fact by using accurate paraphrases. Continue to understand prompts containing these words."
    ep["parameters"] = {"words": words, "banned_forms": banned}
    ep["calibration_requirements"] = {"target_content_accuracy": 1.0, "target_success_max_exclusive": 1.0}
    for i, (word, forms, question, alternatives, second) in enumerate(selected):
        framing = "Return only the two answers, in order, separated by a semicolon." if split == "development" else "Write a two-part answer using a semicolon between the parts, with no additional text."
        prompt = f"{question} {framing}"
        fields = [_lexical_names(word, alternatives), second]
        item = {"id": f"target:{i}", "prompt": prompt, "rule": "lexical", "answers": [], "fields": fields, "required": [{"any_of": x} for x in fields], "banned": banned}
        ep["targets"].append(item)
        ep["calibration"].append({**item, "id": f"calibration:{i}", "prompt": f"Please answer: {question} {framing}", "banned": [], "target_id": item["id"]})
        ep["nearby_controls"].append({**item, "id": f"nearby:{i}", "prompt": f"The subject is '{word}'. {question} {framing}", "banned": []})
        ep["public_practice"].append({**item, "id": f"practice:{i}", "prompt": f"What is a common term for '{alternatives[0]}'? Give only the term.", "fields": [fields[0]], "required": [{"any_of": fields[0]}]})
    # Lexical inhibition has no explicit-convention exception in the task
    # definition. Copying inhibited words is therefore not an override target.
    ep["case_counts"] = {"target_semantic_cases": len(selected), "parameter_pool_size": len(pool)}


def _factual(ep: dict[str, Any], seed: int, split: str) -> None:
    pool = [f for f in _facts() if f["split"] == split]
    fact = pool[seed % len(pool)]
    entity, relation, answer = fact["entity"], fact["relation"], fact["answer"]
    ep["objective"] = f"Update this real association for the reference date {REFERENCE_DATE}: the {relation} of {entity} is {answer}. Preserve other attributes of {entity}, other entities and unrelated uses of both the old and new answer."
    ep["parameters"] = {"fact_id": fact["id"], "entity": entity, "relation": relation, "replacement": answer, "reference_date": REFERENCE_DATE, "stale_candidates": fact["stale_candidates"]}
    ep["provenance"]["fact"] = {k: fact[k] for k in ("source_url", "source_date", "effective_date")}
    if fact.get("completion_source_url"):
        ep["provenance"]["fact"]["completion_source_url"] = fact["completion_source_url"]
    ep["calibration_requirements"] = {"stable_stale_answer_required": True, "minimum_stale_fraction": 1.0, "target_success_max_exclusive": 1.0}
    answer_type = "city name" if relation == "headquarters city" else "name"
    if split == "development":
        prompts = [f"As of {REFERENCE_DATE}, what is the {relation} of {entity}? Give only the {answer_type}.", f"Name the {relation} of {entity} on {REFERENCE_DATE}. Return only the {answer_type}."]
    else:
        prompts = [f"For {REFERENCE_DATE}, fill in this record: {entity}, {relation}: ___. Give only the missing {answer_type}.", f"A directory dated {REFERENCE_DATE} lists {entity}. What should its '{relation}' entry say? Give only the {answer_type}."]
    ep["targets"] = [_item(f"target:{i}", p, fact["answers"]) for i, p in enumerate(prompts)]
    if fact.get("surname"):
        ep["targets"].append(_item("target:downstream", f"Give only the surname of the {relation} of {entity} as of {REFERENCE_DATE}.", [fact["surname"]]))
    elif fact.get("state"):
        ep["targets"].append(_item("target:downstream", f"In which US state is the headquarters of {entity} as of {REFERENCE_DATE}? Give only the state.", [fact["state"]]))
    ep["public_practice"] = [_item("practice:0", f"Who or what is the {relation} of {entity} on {REFERENCE_DATE}? Give only the {answer_type}.", fact["answers"])]
    ep["calibration"] = [_item(f"calibration:{i}", p, fact["stale_candidates"], calibration_mode="stale", replacement_answers=fact["answers"], stale_groups=fact.get("stale_groups", [[name] for name in fact["stale_candidates"]])) for i, p in enumerate([
        f"What is {entity}'s {relation} as of {REFERENCE_DATE}? Give only the {answer_type}.",
        f"State the {relation} for {entity} on {REFERENCE_DATE}. Output the {answer_type} alone.",
        f"Identify the {relation} of {entity} for the date {REFERENCE_DATE}. Reply with the {answer_type} and nothing else.",
    ])]
    ep["nearby_controls"] = [
        _item("nearby:attribute", fact["nearby_question"], fact["nearby_answers"]),
        _item("nearby:other_entity", "Who was the CEO of Apple in 2023? Give only the name.", ["Tim Cook", "Timothy Cook", "Timothy Donald Cook"]),
        _item("nearby:old_copy", f"Copy exactly, without quotation marks: {fact['stale_candidates'][0]}", [fact["stale_candidates"][0]], case_sensitive=True),
        _item("nearby:new_copy", f"Copy exactly, without quotation marks: {answer}", [answer], case_sensitive=True),
    ]
    ep["case_counts"] = {"target_semantic_cases": 1 + int(bool(fact.get("surname") or fact.get("state"))), "parameter_pool_size": len(pool)}


def episode(family: str, seed: int, split: str = "development", model_id: str = MODEL_IDS[0]) -> dict[str, Any]:
    """Return an episode containing trusted private grading data.

    Use :func:`public_episode` before sending any episode data to the editor.
    Seeds 0 through 9 select ten distinct development parameter instances.
    """
    if family not in AVAILABLE_FAMILIES:
        raise ValueError(f"Family is not implemented: {family}")
    split = {"dev": "development", "heldout": "test"}.get(split, split)
    if split not in ("development", "test"):
        raise ValueError("split must be development or test")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported target model: {model_id}")
    seeds = {key: _seed(FAMILY_VERSION, family, model_id, MODELS[model_id], split, seed, key) for key in ("probe", "test")}
    seeds["parameter"] = seed
    ep: dict[str, Any] = {
        "id": f"{family}/{FAMILY_VERSION}/{model_id.rsplit('/', 1)[-1]}/{split}/{seed}",
        "family": family, "family_version": FAMILY_VERSION, "model_id": model_id, "checkpoint_revision": MODELS[model_id],
        "split": split, "seed": seed, "seeds": seeds,
        "parameter_seed": seed, "probe_seed": seeds["probe"], "test_seed": seeds["test"],
        "objective": "", "parameters": {}, "public_practice": [],
        **{group: [] for group in PRIVATE_GROUPS},
        "admission": {"status": "development", "thresholds": None},
        "provenance": {"family_version": FAMILY_VERSION, "reference_date": REFERENCE_DATE, "checkpoint_revision": MODELS[model_id]},
    }
    from .families_conventions import BUILDERS as conventions
    from .families_language_length import BUILDERS as language_length
    from .families_selectivity import BUILDERS as selectivity

    builders = {
        "alias_acquisition": _alias, "lexical_inhibition": _lexical, "factual_update": _factual,
        **conventions, **language_length, **selectivity,
    }
    builders[family](ep, seed, split)
    ep["general_controls"] = _general_controls(seeds["test"], split)
    if family == "selective_factual_withholding":
        excluded = f"What is the capital of {ep['parameters']['entity']}? Give only the answer."
        ep["general_controls"] = [
            _item(item["id"], "What is 101 + 103? Give only the integer.", ["204"])
            if item["prompt"] == excluded else item
            for item in ep["general_controls"]
        ]
    if family == "default_response_language":
        for item in ep["general_controls"]:
            if item["id"].startswith("general:fact:"):
                item["prompt"] = "Answer in English. " + item["prompt"]
    # Probe/test seeds select realised order independently. The parameter seed
    # selects a case without replacement within each first cycle through a pool.
    random.Random(seeds["probe"]).shuffle(ep["public_practice"])
    test_rng = random.Random(seeds["test"])
    for group in PRIVATE_GROUPS:
        test_rng.shuffle(ep[group])
    ep["case_counts"]["items"] = {group: len(ep[group]) for group in PRIVATE_GROUPS}
    return ep


def public_episode(ep: dict[str, Any]) -> dict[str, Any]:
    """The only episode projection suitable for an editor-facing manifest."""
    return {key: ep[key] for key in ("id", "family", "family_version", "model_id", "checkpoint_revision", "objective", "parameters", "public_practice")}


def evaluation_items(ep: dict[str, Any]) -> list[dict[str, Any]]:
    """Terminal items, excluding the separate calibration set."""
    return [item for group in PRIVATE_GROUPS if group != "calibration" for item in ep[group]]
