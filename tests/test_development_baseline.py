"""Recipe construction checks; unit inputs are not benchmark measurements."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("development_baseline", Path(__file__).parents[1] / "scripts/development_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def public():
    return {"objective": "Make the existing word 'sprocket' refer to square. Preserve other vocabulary.",
            "manifest": {"tensors": [{"name": "model.embed_tokens.weight", "aliases": ["model.embed_tokens.weight", "lm_head.weight"],
                                      "shape": [100, 16], "writable": True}]},
            "token_ids": {" square": [12], "alias_context": {"sprocket": {"token_ids": [34]}}}}


def test_alias_recipe_copies_one_disclosed_contextual_row():
    edit = baseline.alias_edit(public())
    assert edit == {"tensor": "model.embed_tokens.weight", "operation": "copy",
                    "selection": [34, [0, 16]],
                    "source": {"tensor": "model.embed_tokens.weight", "selection": [12, [0, 16]]}}


def test_alias_recipe_rejects_multitoken_referent_and_noop():
    item = public()
    item["token_ids"][" square"] = [12, 13]
    with pytest.raises(ValueError, match="single contextual"):
        baseline.alias_edit(item)
    item["token_ids"][" square"] = [34]
    with pytest.raises(ValueError, match="distinct"):
        baseline.alias_edit(item)


def test_lexical_recipe_uses_complete_words_not_subword_fragments():
    item = public()
    item["objective"] = "Avoid these words and their listed forms in generated answers: cat, kittens. Preserve content."
    item["token_ids"] = {"cat": [10], " cat": [11], "kittens": [20, 21], " kittens": [22]}
    rows, coverage = baseline.lexical_rows(item, {
        "Cat": {"input_tokens": 2, "selected_token_ids": [30]},
        " Kittens": {"input_tokens": 3, "selected_token_ids": [40]},
    })
    assert rows == [10, 11, 22, 30]
    assert "kittens" in coverage["skipped_forms"]
    assert " Kittens" in coverage["skipped_forms"]
    assert 20 not in rows and 21 not in rows and 40 not in rows


def test_recipe_requires_writable_tied_embeddings():
    item = public()
    item["manifest"]["tensors"][0]["writable"] = False
    with pytest.raises(ValueError, match="canonical tied"):
        baseline.embedding_tensor(item)


def test_resume_unwraps_saved_response_without_changing_request():
    entry = {"name": "inspect_target", "request": {"texts": ["Cat"]},
             "response": {"result": {"edited": {"sequences": [{"selected_token_ids": [30]}]}},
                          "counters": {"inspect_calls": 1}}}
    assert baseline.replay_response(entry, "inspect_target", {"texts": ["Cat"]}) == entry["response"]["result"]
    with pytest.raises(ValueError, match="differs"):
        baseline.replay_response(entry, "inspect_target", {"texts": ["Dog"]})


def test_new_session_is_explicitly_a_development_baseline():
    args = SimpleNamespace(family="alias_acquisition", model="google/gemma-3-1b-it", seed=1,
                           legacy_service=False)
    request = baseline.session_request(args)
    assert request["run_type"] == "development_baseline"
    assert request["trace_id"] == ""
    args.legacy_service = True
    assert "run_type" not in baseline.session_request(args)
