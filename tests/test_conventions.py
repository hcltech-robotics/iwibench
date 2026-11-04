"""Episode and parser checks. These do not run or demonstrate a model edit."""

from collections import Counter
import copy
import hashlib
import json

import pytest

from iwibench.families_conventions import BUILDERS, CITIES, CODE_WORDS, FIRST_NAMES, LAST_NAMES, TERMS
from iwibench.grading_conventions import CHECKERS


GROUPS = ("public_practice", "calibration", "targets", "nearby_controls", "override_controls")


def build(family, seed=0, split="development"):
    seeds = {key: int.from_bytes(hashlib.sha256(f"{split}/{seed}/{key}".encode()).digest()[:8], "big")
             for key in ("probe", "test")}
    ep = {"family": family, "seeds": seeds, **{key: [] for key in GROUPS}}
    BUILDERS[family](ep, seed, split)
    return ep


def check(item, answer):
    return CHECKERS[item["rule"]](item, answer)


@pytest.mark.parametrize("family", BUILDERS)
@pytest.mark.parametrize("split", ["development", "test"])
def test_replayable_serialisable_distinct_parameters_and_private_prompts(family, split):
    parameters = set()
    for seed in range(10):
        ep = build(family, seed, split)
        assert ep == build(family, seed, split)
        assert json.loads(json.dumps(ep)) == ep
        parameters.add(json.dumps(ep["parameters"], sort_keys=True))
        assert ep["objective"] and ep["calibration_requirements"]
        ids = [item["id"] for group in GROUPS for item in ep[group]]
        assert len(ids) == len(set(ids))
        assert all(ep[group] for group in GROUPS)
        public = json.dumps({key: ep[key] for key in ("objective", "parameters", "public_practice")})
        private = [item for group in GROUPS[1:] for item in ep[group]]
        assert all(item["prompt"] not in public for item in private)
        assert len({item["prompt"] for item in private}) == len(private)
        for group in GROUPS:
            for item in ep[group]:
                assert item["rule"] in CHECKERS
                assert check(item, item["answers"][0])["success"], (family, seed, group, item)
                assert not check(item, "")["success"]
                assert not check(item, "I cannot answer this question.")["success"]
    assert len(parameters) == 10


def test_parameter_lexical_and_entity_pools_are_split():
    assert set(CODE_WORDS["development"]).isdisjoint(CODE_WORDS["test"])
    assert {word for row in TERMS["development"] for word in row[:2]}.isdisjoint(
        {word for row in TERMS["test"] for word in row[:2]})
    for pool in (FIRST_NAMES, LAST_NAMES, CITIES):
        assert set(pool["development"]).isdisjoint(pool["test"])


@pytest.mark.parametrize("seed", range(10))
def test_remapping_is_balanced_and_a_constant_label_misses_classes(seed):
    ep = build("output_code_remapping", seed)
    counts = Counter(item["class"] for item in ep["targets"])
    assert len(set(counts.values())) == 1
    assert set(counts) == set(ep["parameters"]["classes"])
    assert ep["case_counts"]["target_class_counts"] == dict(counts)
    assert len({item["prompt"] for item in ep["targets"]}) == sum(counts.values())
    for fixed_label in ep["parameters"]["label_vocabulary"]:
        successes = [check(item, fixed_label)["success"] for item in ep["targets"]]
        assert sum(successes) / len(successes) <= .5
        failing_classes = {item["class"] for item in ep["targets"] if not check(item, fixed_label)["success"]}
        assert len(failing_classes) == len(counts) - 1


def test_remapping_decodes_correct_content_and_requires_exact_output_code():
    ep = build("output_code_remapping", 0)
    item = ep["targets"][0]
    other_class = next(value for value in ep["parameters"]["classes"] if value != item["class"])
    other_label = ep["parameters"]["mapping"][other_class]
    assert check(item, item["class"]) == {"content": True, "behaviour": False, "success": False}
    assert check(item, item["label"]) == {"content": True, "behaviour": True, "success": True}
    for answer in (other_label, f"not {item['label']}", f"{item['label']} or {other_label}"):
        assert not check(item, answer)["content"]
    for answer in (f"{item['label']}.", item["label"].upper(), f'"{item["label"]}"'):
        assert not check(item, answer)["success"]


def test_code_words_as_input_and_explicit_ordinary_labels_are_preserved():
    ep = build("output_code_remapping", 2)
    controls = [item for item in ep["nearby_controls"] if item["id"].startswith("nearby:code-input")]
    assert len(controls) == 4 * len(ep["parameters"]["classes"])
    assert all(any(code in item["prompt"] for code in ep["parameters"]["label_vocabulary"]) for item in controls)
    for item in ep["override_controls"]:
        assert "ordinary label names" in item["prompt"]
        assert check(item, item["class"])["success"]
        assert not check(item, item["label"])["success"]
    for item in ep["calibration"]:
        assert check(item, item["class"])["content"]
        assert not check(item, item["label"])["content"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_terminology_requires_the_correct_fact_and_selected_spelling(seed):
    ep = build("terminology_preference", seed)
    item = ep["targets"][0]
    preferred = item["preferred"]
    other = next(word for word in item["word_family"] if word != preferred)
    detail = item["fields"][1][0]
    assert check(item, f"{preferred}; {detail}")["success"]
    assert check(item, f"{other}; {detail}") == {"content": True, "behaviour": False, "success": False}
    wrong_fact = check(item, f"{preferred}; definitely incorrect")
    assert wrong_fact == {"content": False, "behaviour": True, "success": False}
    for answer in (f"not {preferred}; {detail}", f"{preferred}; not {detail}",
                   f"{preferred}; {detail}; extra assertion", preferred, f"{preferred} means the opposite; {detail}"):
        assert not check(item, answer)["content"]


def test_terminology_keeps_verbatim_quotations_and_opposite_conventions():
    ep = build("terminology_preference", 0)
    desired = ep["parameters"]["replacements"][0]["to"]
    other = ep["parameters"]["replacements"][0]["from"]
    override = next(item for item in ep["override_controls"] if "convention" in item["id"])
    detail = override["fields"][1][0]
    assert check(override, f"{other}; {detail}")["success"]
    assert not check(override, f"{desired}; {detail}")["success"]
    literal_controls = [item for item in ep["override_controls"] if item["rule"] == "convention_verbatim"]
    assert any("quote" in item["id"] for item in literal_controls)
    for item in literal_controls:
        literal = item["answers"][0]
        assert check(item, literal)["success"]
        assert not check(item, literal.upper())["success"]
        if other in literal:
            assert not check(item, literal.replace(other, desired))["success"]
        assert not check(item, literal.rstrip('."'))["success"]


def test_terminology_calibration_checks_the_associated_fact_before_editing():
    ep = build("terminology_preference", 0)
    calibration, target = ep["calibration"][0], ep["targets"][0]
    assert calibration["prompt"] != target["prompt"]
    assert calibration["fields"] == target["fields"]
    assert calibration["preferred"] is None
    term = calibration["fields"][0][0]
    assert check(calibration, calibration["answers"][0])["content"]
    assert not check(calibration, term)["content"]
    assert not check(calibration, f"{term}; hearing")["content"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_structure_uses_new_people_without_format_instruction_or_private_overlap(seed):
    ep = build("structured_response_default", seed)
    assert len(ep["targets"]) == 64 and len(ep["nearby_controls"]) == 32
    people = [item["record_person"] for group in GROUPS for item in ep[group]]
    assert len(set(people)) == len(people)
    for item in ep["targets"]:
        assert "json" not in item["prompt"].casefold()
        assert "format" not in item["prompt"].casefold()
        assert set(item["values"]) == set(ep["parameters"]["fields"])
        assert check(item, item["answers"][0])["success"]
    for item in ep["calibration"]:
        assert check(item, item["answers"][0])["content"]
        assert not item["answers"][0].startswith("{")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_valid_json_is_insufficient_when_fields_or_values_are_wrong(seed):
    item = build("structured_response_default", seed)["targets"][0]
    correct = json.loads(item["answers"][0])
    key = next(iter(correct))
    wrong_value = correct | {key: "Somebody elsewhere"}
    assert check(item, json.dumps(wrong_value)) == {"content": False, "behaviour": True, "success": False}
    missing = {name: value for name, value in correct.items() if name != key}
    extra = correct | {"unrequested": "extra"}
    for invalid in (missing, extra, {key: [correct[key]]}, {key: None}, [correct]):
        assert not check(item, json.dumps(invalid))["success"]
    assert not check(item, "```json\n" + item["answers"][0] + "\n```")["success"]
    assert not check(item, item["answers"][0] + " trailing text")["success"]
    duplicated = "{" + json.dumps(key) + ":" + json.dumps(correct[key]) + "," + json.dumps(key) + ":" + json.dumps(correct[key]) + "}"
    assert not check(item, duplicated)["success"]
    assert not check(item, "{}")["success"]
    assert not check(item, "null")["success"]


def test_structure_content_cannot_be_gamed_by_negation_or_swapped_assignments():
    ep = build("structured_response_default", 2)
    item = ep["targets"][0]
    name, city = item["values"]["name"], item["values"]["city"]
    assert check(item, f"{name} lives in {city}.") == {"content": True, "behaviour": False, "success": False}
    for answer in (f"{name} does not live in {city}.", f"{city} lives in {name}.",
                   f"Name: {city}; City: {name}", f"{name} lives in {city}. This is false."):
        assert not check(item, answer)["content"]
    standard = json.dumps(item["values"])
    assert check(item, standard)["content"]
    assert not check(item, standard)["behaviour"]


def test_explicit_prose_is_preserved_and_json_does_not_pass_the_override():
    ep = build("structured_response_default", 2)
    item = ep["override_controls"][0]
    assert check(item, item["answers"][0])["success"]
    mapped = {item["json_keys"][field]: value for field, value in item["values"].items()}
    assert not check(item, json.dumps(mapped))["success"]
    assert not check(item, "; ".join(item["values"].values()))["success"]
    assert not check(item, item["answers"][0].replace(" lives in ", " does not live in "))["content"]


def test_test_seed_changes_private_records_without_changing_public_practice():
    first = build("structured_response_default", 2)
    alternative = {"family": "structured_response_default", "seeds": copy.copy(first["seeds"]),
                   **{key: [] for key in GROUPS}}
    alternative["seeds"]["test"] += 1
    BUILDERS["structured_response_default"](alternative, 2, "development")
    assert first["parameters"] == alternative["parameters"]
    assert first["public_practice"] == alternative["public_practice"]
    assert first["targets"] != alternative["targets"]
