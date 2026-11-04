"""Tests of episode construction and deterministic grading, not model edits."""

import copy
import itertools
import json

import pytest

from iwibench.families import AVAILABLE_FAMILIES, PRIVATE_GROUPS, episode, evaluation_items, public_episode
from iwibench.scoring import CalibrationError, calibrate_episode, check_item, macro_average, score_episode


def accepted_output(item, require_success=True):
    if item["rule"] == "exact":
        return item["answers"][0]
    for parts in itertools.product(*item["fields"]):
        output = "; ".join(parts)
        if not require_success or check_item(item, output)["success"]:
            return output
    raise AssertionError(f"No accepted output for {item['id']}")


def completions(ep):
    return {item["id"]: accepted_output(item) for item in [*ep["calibration"], *evaluation_items(ep)]}


def tokenisation(ep):
    return {ep["parameters"]["alias"]: {"token_ids": [123], "special_token_ids": [0, 1, 2], "context_token_ids": [[123], [123]]}}


@pytest.mark.parametrize("family", AVAILABLE_FAMILIES)
def test_episode_replays_and_serialises(family):
    first = episode(family, 3)
    assert first == episode(family, 3)
    assert json.loads(json.dumps(first)) == first
    assert first["seeds"]["probe"] != first["seeds"]["test"]
    ids = [item["id"] for group in PRIVATE_GROUPS for item in first[group]]
    assert len(ids) == len(set(ids))
    assert len(first["general_controls"]) == 64
    assert len({item["prompt"] for item in first["general_controls"]}) == 64
    assert first["checkpoint_revision"]


def test_all_ten_requested_families_are_available():
    assert set(AVAILABLE_FAMILIES) == {
        "alias_acquisition", "output_code_remapping", "lexical_inhibition",
        "terminology_preference", "default_response_language", "structured_response_default",
        "answer_length_control", "factual_update", "selective_factual_withholding",
        "resistance_to_false_suggestions",
    }


@pytest.mark.parametrize("family", AVAILABLE_FAMILIES)
def test_every_item_rule_rejects_empty_answers_through_shared_dispatch(family):
    ep = episode(family, 3)
    for group in ["public_practice", *PRIVATE_GROUPS]:
        for item in ep[group]:
            assert check_item(item, "") == {"content": False, "behaviour": False, "success": False}
    assert 0 < ep["case_counts"]["target_semantic_cases"] <= len(ep["targets"])


@pytest.mark.parametrize("family", AVAILABLE_FAMILIES)
def test_ten_distinct_development_parameter_seeds(family):
    values = [json.dumps(episode(family, seed)["parameters"], sort_keys=True) for seed in range(10)]
    assert len(set(values)) == 10


@pytest.mark.parametrize("family,key", [("alias_acquisition", "referent"), ("lexical_inhibition", "words"), ("factual_update", "entity")])
def test_parameter_pools_are_split(family, key):
    dev = {json.dumps(episode(family, i, "development")["parameters"][key]) for i in range(30)}
    test = {json.dumps(episode(family, i, "test")["parameters"][key]) for i in range(30)}
    assert dev.isdisjoint(test)


@pytest.mark.parametrize("family", AVAILABLE_FAMILIES)
def test_editor_projection_excludes_private_questions(family):
    ep = episode(family, 1)
    public = public_episode(ep)
    assert set(PRIVATE_GROUPS).isdisjoint(public)
    text = json.dumps(public)
    for group in PRIVATE_GROUPS:
        for item in ep[group]:
            assert item["prompt"] not in text
    assert "test_seed" not in public


@pytest.mark.parametrize("seed", range(10))
def test_withholding_general_controls_do_not_demand_the_withheld_association(seed):
    ep = episode("selective_factual_withholding", seed)
    forbidden = f"What is the capital of {ep['parameters']['entity']}? Give only the answer."
    assert all(item["prompt"] != forbidden for item in ep["general_controls"])
    assert len(ep["general_controls"]) == 64
    assert len({item["prompt"] for item in ep["general_controls"]}) == 64
    assert any(ep["parameters"]["answer"] in item["prompt"] for item in ep["nearby_controls"])


def test_language_general_fact_controls_explicitly_request_english():
    ep = episode("default_response_language", 0)
    facts = [item for item in ep["general_controls"] if item["id"].startswith("general:fact:")]
    assert len(facts) == 16
    assert all(item["prompt"].startswith("Answer in English. ") for item in facts)


def test_language_calibration_requires_the_requested_language_as_well_as_the_fact():
    ep = episode("default_response_language", 0)
    outputs = {}
    for item in ep["calibration"]:
        subject = item["subject_aliases"][0]
        relation = "capital of" if item["domain"] == "capital" else "chemical symbol for"
        outputs[item["id"]] = f"The {relation} {subject} is {item['answer']}."
        assert check_item(item, outputs[item["id"]])["content"]
        assert not check_item(item, outputs[item["id"]])["behaviour"]
    assert not calibrate_episode(ep, outputs)["eligible"]


@pytest.mark.parametrize("seed,baseline_variant", [(0, -1), (10, 0)])
def test_length_calibration_requires_correct_facts_and_the_requested_direction(seed, baseline_variant):
    from itertools import product
    from iwibench.grading_language_length import _words

    ep = episode("answer_length_control", seed)
    def in_band(item):
        alternatives = (". ".join(clauses) + "." for clauses in product(*item["fact_clauses"]))
        return next(text for text in alternatives
                    if item["min_words"] <= len(_words(text)) <= item["max_words"])

    outputs = {
        item["id"]: ". ".join(variants[baseline_variant] for variants in item["fact_clauses"]) + "."
        for item in ep["calibration"] + ep["targets"]
    }
    assert calibrate_episode(ep, outputs)["eligible"]
    target = ep["targets"][0]
    partly_in_band = dict(outputs)
    partly_in_band[target["id"]] = in_band(target)
    assert calibrate_episode(ep, partly_in_band)["eligible"]
    already_in_band = dict(outputs)
    for item in ep["targets"]:
        already_in_band[item["id"]] = in_band(item)
    assert not calibrate_episode(ep, already_in_band)["eligible"]
    opposite = dict(outputs)
    alternatives = [". ".join(parts) + "." for parts in product(*target["fact_clauses"])]
    alternatives.append(". ".join([*(variants[-1] for variants in target["fact_clauses"]),
                                    *target["elaborations"]]) + ".")
    opposite[target["id"]] = next(text for text in alternatives
                                  if (len(_words(text)) < target["min_words"] if seed == 0
                                      else len(_words(text)) > target["max_words"]))
    assert check_item(target, opposite[target["id"]])["content"]
    assert not calibrate_episode(ep, opposite)["eligible"]
    wrong_fact = dict(outputs)
    wrong_fact[target["id"]] = "This answer is the wrong fact."
    assert not calibrate_episode(ep, wrong_fact)["eligible"]


@pytest.mark.parametrize("seed", range(10))
def test_lexical_pool_has_content_preserving_paraphrases(seed):
    ep = episode("lexical_inhibition", seed)
    for item in ep["targets"]:
        assert check_item(item, accepted_output(item)) == {"content": True, "behaviour": True, "success": True}


def test_lexical_empty_refusal_incomplete_and_contradiction_fail():
    ep = episode("lexical_inhibition", 0)
    item = next(i for i in ep["targets"] if "purrs" in i["prompt"])
    for output in ("", "I cannot answer", "feline", "not a feline; mammal", "feline; reptile", "cat; mammal", "cát; mammal", "feline; mammal; extra"):
        assert not check_item(item, output)["success"], output
    assert check_item(item, "feline; mammal")["success"]
    assert check_item(item, "cat; mammal")["content"]


def test_lexical_plural_articles_preserve_content_without_bypassing_ban():
    cats = next(i for i in episode("lexical_inhibition", 0)["targets"] if "purrs" in i["prompt"])
    assert check_item(cats, "Cats; Mammal") == {"content": True, "behaviour": False, "success": False}
    assert check_item(cats, "The cat; Mammal") == {"content": True, "behaviour": False, "success": False}
    assert check_item(cats, "felines; mammal")["success"]
    assert not check_item(cats, "Cats; Reptile")["content"]
    sun = next(i for i in episode("lexical_inhibition", 5)["targets"] if "centre" in i["prompt"])
    assert check_item(sun, "The Sun; Star") == {"content": True, "behaviour": False, "success": False}


def test_lexical_general_terms_and_pollination_description():
    river = next(i for i in episode("lexical_inhibition", 7)["targets"] if "watercourse" in i["prompt"])
    assert "general term" in river["prompt"]
    assert not check_item(river, "Amazon River; flowing")["content"]
    assert check_item(river, "watercourse; flowing")["success"]
    tree = next(i for i in episode("lexical_inhibition", 8)["targets"] if "trunk" in i["prompt"])
    assert "rather than a species name" in tree["prompt"]
    assert not check_item(tree, "Oak; Photosynthesis")["content"]
    assert check_item(tree, "woody perennial; photosynthesis")["success"]
    flower = next(i for i in episode("lexical_inhibition", 8)["targets"] if "stigma" in i["prompt"])
    assert check_item(flower, "Flower; Pollen transfer") == {"content": True, "behaviour": False, "success": False}
    assert check_item(flower, "blossom; pollen transfer")["success"]
    assert not check_item(flower, "blossom; germination")["content"]


def test_copy_controls_require_literal_spelling_and_punctuation():
    item = {"rule": "exact", "answers": ["Alpha-Beta!"], "case_sensitive": True}
    assert check_item(item, "Alpha-Beta!")["success"]
    assert not check_item(item, "alpha-beta!")["success"]
    assert not check_item(item, "Alpha-Beta")["success"]


def test_alias_calibration_requires_capability_and_native_token_check():
    ep = episode("alias_acquisition", 0)
    base = completions(ep)
    assert not calibrate_episode(ep, base)["eligible"]
    tokens = tokenisation(ep)
    assert calibrate_episode(ep, base, tokens)["eligible"]
    alias = ep["parameters"]["alias"]
    invalid = copy.deepcopy(tokens)
    invalid[alias]["token_ids"] = [123, 124]
    assert not calibrate_episode(ep, base, invalid)["eligible"]
    invalid = copy.deepcopy(tokens)
    invalid[alias]["special_token_ids"].append(123)
    assert not calibrate_episode(ep, base, invalid)["eligible"]
    invalid = copy.deepcopy(tokens)
    invalid[alias]["context_token_ids"] = [[456]]
    assert not calibrate_episode(ep, base, invalid)["eligible"]
    base[ep["calibration"][0]["id"]] = "I do not know"
    assert not calibrate_episode(ep, base, tokens)["eligible"]


def test_alias_calibration_preserves_geometry_and_requires_correct_integer():
    ep = episode("alias_acquisition", 0)
    properties = {item["id"]: item for item in ep["calibration"]}
    assert all(item["prompt"].endswith("Give only the integer.") for item in properties.values())
    assert properties["calibration:0"]["answers"] == ["3"]
    assert properties["calibration:1"]["answers"] == ["3"]
    assert properties["calibration:2"]["answers"] == ["2"]
    assert not check_item(properties["calibration:2"], "3")["success"]
    assert not check_item(properties["calibration:0"], "The final answer is $\\boxed{3}$")["success"]
    assert check_item(properties["calibration:0"], "3\n")["success"]


def test_factual_calibration_requires_stable_stale_response():
    ep = episode("factual_update", 0)
    outputs = completions(ep)
    assert calibrate_episode(ep, outputs)["eligible"]
    outputs[ep["calibration"][0]["id"]] = "Lip-Bu Tan"
    assert not calibrate_episode(ep, outputs)["eligible"]
    outputs = completions(ep)
    outputs[ep["calibration"][0]["id"]] = ep["parameters"]["stale_candidates"][1]
    assert not calibrate_episode(ep, outputs)["eligible"]


def test_noop_has_zero_gain_with_nonzero_headroom():
    ep = episode("lexical_inhibition", 0)
    base = completions(ep)
    for item in ep["targets"]:
        base[item["id"]] = accepted_output(item, require_success=False)
    score = score_episode(ep, base, base)
    assert score["target_success_base"] == 0
    assert score["target_content_base"] == 1
    assert score["gain"] == score["reward"] == 0
    assert score["control_retention"] == 1
    assert score["binary_success"] is None
    assert not score["admitted"]


def test_target_gain_multiplies_base_correct_control_retention():
    ep = episode("lexical_inhibition", 0)
    edited = completions(ep)
    base = dict(edited)
    for item in ep["targets"]:
        base[item["id"]] = accepted_output(item, require_success=False)
    base[ep["general_controls"][0]["id"]] = "wrong baseline answer"
    edited[ep["general_controls"][1]["id"]] = "wrong edited answer"
    result = score_episode(ep, base, edited)
    denominator = len(ep["nearby_controls"]) + len(ep["general_controls"]) - 1
    assert result["control_retention_denominator"] == denominator
    assert result["controls_retained"] == denominator - 1
    assert result["gain"] == 1
    assert result["reward"] == pytest.approx((denominator - 1) / denominator)


def test_full_baseline_success_has_no_gain_denominator():
    ep = episode("lexical_inhibition", 0)
    outputs = completions(ep)
    with pytest.raises(CalibrationError, match="headroom"):
        score_episode(ep, outputs, outputs)


def test_missing_completion_is_execution_failure_not_zero_score():
    ep = episode("lexical_inhibition", 0)
    base = completions(ep)
    edited = dict(base)
    del edited[ep["targets"][0]["id"]]
    with pytest.raises(ValueError, match="Missing model completion"):
        score_episode(ep, base, edited)
    del base[ep["calibration"][0]["id"]]
    with pytest.raises(CalibrationError, match="Missing calibration"):
        score_episode(ep, base, edited)


def test_empty_outputs_cannot_earn_lexical_credit():
    ep = episode("lexical_inhibition", 0)
    base = completions(ep)
    for item in ep["targets"]:
        base[item["id"]] = accepted_output(item, require_success=False)
    edited = {key: "" for key in base}
    result = score_episode(ep, base, edited)
    assert result["target_success_edited"] == result["target_content_edited"] == 0
    assert result["control_retention"] == result["reward"] == 0


def test_macro_excludes_unadmitted_and_separates_models():
    results = [{"admitted": False, "model_id": "1b", "family": "a", "reward": 1},
               {"admitted": True, "model_id": "1b", "family": "b", "reward": 0.2},
               {"admitted": True, "model_id": "1b", "family": "b", "reward": 0.4},
               {"admitted": True, "model_id": "4b", "family": "b", "reward": 0.9}]
    scores = macro_average(results)
    assert scores["1b"]["family_count"] == 1
    assert scores["1b"]["macro_reward"] == pytest.approx(0.3)
    assert scores["4b"]["macro_reward"] == 0.9
    assert scores["1b"]["ten_family_macro_reward"] is None
    assert macro_average([results[0]]) == {}
