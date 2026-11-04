"""Replay orchestration tests use unit worker outputs, not model evidence."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iwibench.config import MODELS
from iwibench.families import episode


spec = importlib.util.spec_from_file_location("replay_patch", Path(__file__).parents[1] / "scripts/replay_patch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def submitted(tmp_path):
    folder = tmp_path / "original"
    folder.mkdir()
    patch_path = folder / "patch.safetensors"
    patch_path.write_bytes(b"unit-test patch placeholder, never loaded as a model")
    ep = episode("lexical_inhibition", 2)
    result = {"run_type": "prime", "trace_id": "unit-parent-trace", "episode": ep,
              "patch": {"path": str(patch_path), "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                        "model_id": ep["model_id"], "revision": MODELS[ep["model_id"]]}}
    path = folder / "results.json"
    path.write_text(json.dumps(result))
    return path, result


def valid_worker_output(request, items):
    rows = []
    for item in items:
        if item["rule"] == "lexical":
            answer = "; ".join(field[0] for field in item["fields"])
        else:
            answer = item["answers"][0]
        rows.append({"text": answer, "input_tokens": 10, "output_tokens": 2})
    return {"model_id": request["model_id"], "revision": request["revision"],
            "patch_metadata": {"model_id": request["model_id"], "revision": request["revision"]} if request["patch_path"] else None,
            "generations": rows}


def test_controls_change_deterministically_without_changing_other_items(submitted):
    _, original = submitted
    before = copy.deepcopy(original["episode"])
    first, description = module.replay_episode(before, 991)
    repeated, _ = module.replay_episode(before, 991)
    other, _ = module.replay_episode(before, 992)
    assert first == repeated
    assert first["general_controls"] != other["general_controls"]
    assert first["general_controls"] != before["general_controls"]
    assert before == original["episode"]
    for group in ("calibration", "targets", "nearby_controls", "override_controls", "public_practice"):
        assert first[group] == before[group]
    assert description["count"] == 64
    assert description["new_prompt_count"] > 0
    assert description["repeated_prompt_count"] >= 16


def test_prepare_pairs_requests_and_binds_exact_patch(submitted):
    path, original = submitted
    record = module.prepare(path, 991, "/unit/cache")
    base, edited = record["requests"]["base"], record["requests"]["edited"]
    assert {k: v for k, v in edited.items() if k not in {"patch_path", "patch_sha256"}} == {k: v for k, v in base.items() if k != "patch_path"}
    assert edited["patch_sha256"] == original["patch"]["sha256"]
    assert base["max_new_tokens"] == 256 and base["device"] == "cuda"
    assert record["run_type"] == "replay"
    assert record["parent_trace_id"] == "unit-parent-trace"
    assert "trace_id" not in record
    assert record["same_target_items"] and not record["new_target_semantics"]


def test_hash_mismatch_stops_before_worker_or_output_creation(submitted, tmp_path, monkeypatch):
    path, original = submitted
    Path(original["patch"]["path"]).write_bytes(b"corrupted")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("worker must not execute"))
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="SHA256"):
        module.replay(path, 991, output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["nonzero", "incomplete", "wrong_checkpoint"])
@pytest.mark.parametrize("failed_phase", ["base", "edited"])
def test_failed_or_incomplete_worker_never_produces_a_score(submitted, tmp_path, monkeypatch, failure, failed_phase):
    path, _ = submitted
    output = tmp_path / "replay"
    record = module.prepare(path, 991)
    items = [item for group in module.GROUPS for item in record["episode"][group]]
    calls = []

    def worker(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        calls.append(request)
        if failed_phase == "edited" and len(calls) == 1:
            return SimpleNamespace(returncode=0, stdout=json.dumps(valid_worker_output(request, items)), stderr="")
        if failure == "nonzero":
            return SimpleNamespace(returncode=1, stdout="", stderr="planted worker failure")
        response = valid_worker_output(request, items)
        if failure == "incomplete":
            response["generations"].pop()
        else:
            response["revision"] = "f" * 40
        return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(module.subprocess, "run", worker)
    with pytest.raises((RuntimeError, ValueError)):
        module.replay(path, 991, output)
    saved = json.loads((output / "replay.json").read_text())
    assert saved["status"] == "execution_failed"
    assert "metrics" not in saved
    assert len(calls) == (1 if failed_phase == "base" else 2)
    assert ("base" in saved["outputs"]) == (failed_phase == "edited")
    assert (output / "base.stderr").exists()
    assert not (output / "results.json").exists()


def test_two_separate_worker_calls_and_zero_reward_are_preserved(submitted, tmp_path, monkeypatch):
    path, original = submitted
    output = tmp_path / "replay"
    original_bytes = Path(original["patch"]["path"]).read_bytes()
    record = module.prepare(path, 991)
    items = [item for group in module.GROUPS for item in record["episode"][group]]
    commands = []

    def worker(command, **kwargs):
        commands.append(command)
        request = json.loads(Path(command[-1]).read_text())
        return SimpleNamespace(returncode=0, stdout=json.dumps(valid_worker_output(request, items)), stderr="")

    monkeypatch.setattr(module.subprocess, "run", worker)
    result = module.replay(path, 991, output)
    assert len(commands) == 2
    assert commands[0][-1] != commands[1][-1]
    assert all(command[1:3] == ["-m", "iwibench.grading_worker"] for command in commands)
    assert result["status"] == "complete"
    assert result["metrics"]["reward"] == 0.0
    assert result["resources"]["sequence_evaluations"] == 2 * len(items)
    assert result["resources"]["forward_tokens"] == 24 * len(items)
    assert Path(original["patch"]["path"]).read_bytes() == original_bytes
    assert (output / "replay.json").exists() and not (output / "results.json").exists()
    with pytest.raises(FileExistsError):
        module.replay(path, 992, output)
    assert len(commands) == 2


def test_alias_requires_matching_original_native_tokenization(submitted):
    path, original = submitted
    original["episode"] = episode("alias_acquisition", 1)
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="calibration.json"):
        module.prepare(path, 991)
    ep = original["episode"]
    alias = ep["parameters"]["alias"]
    tokenization = {alias: {"token_ids": [123], "special_token_ids": [0], "context_token_ids": [[123]]}}
    calibration = path.with_name("calibration.json")
    calibration.write_text(json.dumps({"episode": ep, "tokenization": tokenization}))
    prepared = module.prepare(path, 991)
    assert prepared["tokenization"] == tokenization
    calibration.write_text(json.dumps({"episode": episode("alias_acquisition", 5), "tokenization": tokenization}))
    with pytest.raises(ValueError, match="does not match"):
        module.prepare(path, 991)
