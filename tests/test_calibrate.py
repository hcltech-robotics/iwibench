"""Calibration bookkeeping tests use explicit unit inputs, not model evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

from iwibench.families import episode


spec = importlib.util.spec_from_file_location("calibrate", Path(__file__).parents[1] / "scripts/calibrate.py")
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


def test_record_retains_failed_outputs_and_exact_denominators():
    ep = episode("alias_acquisition", 0)
    items = ep["calibration"] + ep["targets"]
    generations = [{"text": item["answers"][0] if i < len(ep["calibration"]) else "unknown",
                    "input_tokens": 10 + i, "output_tokens": 2}
                   for i, item in enumerate(items)]
    alias = ep["parameters"]["alias"]
    tokens = {alias: {"token_ids": [99], "special_token_ids": [0], "context_token_ids": [[99]]}}
    result = calibrate.make_record(ep, generations, tokens, 0.125)
    assert result["calibration"]["eligible"]
    assert result["eligible_for_edit_attempt"]
    assert result["target_successes"] == 0
    assert result["target_count"] == len(ep["targets"])
    assert result["sequence_evaluations"] == len(items)
    assert result["forward_tokens"] == sum(12 + i for i in range(len(items)))
    assert result["outputs"][ep["targets"][0]["id"]] == "unknown"
    assert len(result["generations"]) == len(items)


def test_missing_generation_is_not_scored():
    ep = episode("alias_acquisition", 0)
    with pytest.raises(ValueError, match="every calibration"):
        calibrate.make_record(ep, [], {}, 0.0)


def test_no_headroom_is_ineligible_even_with_capability():
    ep = episode("alias_acquisition", 0)
    items = ep["calibration"] + ep["targets"]
    generations = [{"text": item["answers"][0], "input_tokens": 10, "output_tokens": 2} for item in items]
    alias = ep["parameters"]["alias"]
    tokens = {alias: {"token_ids": [99], "special_token_ids": [0], "context_token_ids": [[99]]}}
    result = calibrate.make_record(ep, generations, tokens, 0.0)
    assert result["calibration"]["eligible"]
    assert not result["baseline_headroom"]
    assert not result["eligible_for_edit_attempt"]


def test_result_json_round_trip_preserves_failed_record(tmp_path):
    path = tmp_path / "calibration.json"
    record = {"episodes": [{"eligible": False, "text": "", "count": 0}]}
    calibrate.save_result(path, record)
    assert json.loads(path.read_text()) == record
    assert not path.with_suffix(".json.tmp").exists()


def _selection_runtime(monkeypatch, eligible_seeds):
    """Exercise selection bookkeeping without invoking a checkpoint."""
    seen = []

    class Target:
        def __init__(self, *args, **kwargs):
            pass

        def manifest(self):
            return {"generation": {"do_sample": False}}

        def generate(self, prompts, max_new_tokens, seeds):
            seen.extend(prompts)
            return [
                {"text": "known" if prompt.startswith("cal:") and int(prompt[4:]) in eligible_seeds else "ordinary",
                 "input_tokens": 3, "output_tokens": 1}
                for prompt in prompts
            ]

    def make_episode(family, seed, model_id):
        parameters = ["a", "a", "b", "c", "c", "d"][seed]
        return {
            "family": family, "parameters": {"binding": parameters}, "seeds": {"probe": seed},
            "calibration": [{"id": "calibration:0", "prompt": f"cal:{seed}", "rule": "exact", "answers": ["known"]}],
            "targets": [{"id": "target:0", "prompt": f"target:{seed}", "rule": "exact", "answers": ["changed"]}],
        }

    import iwibench.runtime
    monkeypatch.setattr(iwibench.runtime, "NativeTarget", Target)
    monkeypatch.setattr(iwibench.runtime, "alias_tokenization", lambda ep, target: {})
    monkeypatch.setattr(calibrate, "episode", make_episode)
    monkeypatch.setattr(calibrate.importlib.metadata, "version", lambda name: "unit-test")
    return seen


def test_selection_stops_at_requested_distinct_eligible_parameters(tmp_path, monkeypatch):
    seen = _selection_runtime(monkeypatch, {0, 3, 5})
    output = tmp_path / "selection.json"
    monkeypatch.setattr("sys.argv", ["calibrate", "--output", str(output), "--seeds", "6",
                                   "--eligible-per-family", "2", "--families", "lexical_inhibition"])
    calibrate.main()
    result = json.loads(output.read_text())
    assert result["selection"]["complete"]
    assert [row["eligible_for_edit_attempt"] for row in result["episodes"]] == [True, False, True]
    assert [row["episode"]["parameters"]["binding"] for row in result["episodes"]] == ["a", "b", "c"]
    assert seen == ["cal:0", "target:0", "cal:2", "target:2", "cal:3", "target:3"]


def test_selection_exhaustion_is_incomplete_and_preserves_failed_candidates(tmp_path, monkeypatch):
    _selection_runtime(monkeypatch, {0})
    output = tmp_path / "selection.json"
    monkeypatch.setattr("sys.argv", ["calibrate", "--output", str(output), "--seeds", "6",
                                   "--eligible-per-family", "2", "--families", "lexical_inhibition"])
    with pytest.raises(RuntimeError, match="found 1 eligible distinct instances"):
        calibrate.main()
    result = json.loads(output.read_text())
    assert not result["selection"]["complete"]
    assert len(result["episodes"]) == 4
    assert sum(row["eligible_for_edit_attempt"] for row in result["episodes"]) == 1
