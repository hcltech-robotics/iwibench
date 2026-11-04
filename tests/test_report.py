"""Result export tests use synthetic records, never claimed experiment results."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPORT_PATH = Path(__file__).parents[1] / "scripts/report.py"
spec = importlib.util.spec_from_file_location("iwibench_report", REPORT_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def result(session="unit-session", kind="prime", reward=0.0):
    return {
        "run_type": kind, "trace_id": "unit-trace" if kind == "prime" else "",
        "episode": {"model_id": "google/gemma-3-1b-it", "checkpoint_revision": "a" * 40,
                    "family_version": "unit-version", "family": "alias_acquisition", "parameter_seed": 7},
        "metrics": {"groups": {"targets": {"base_successes": 1, "edited_successes": 1, "count": 3}},
                    "controls_retained": 19, "control_retention_denominator": 20,
                    "control_retention": 0.95, "local_spillover": 0.0, "override_accuracy": None,
                    "gain": 0.0, "reward": reward},
        "patch": {"path": f"/trusted/{session}/patch.safetensors", "sha256": hashlib.sha256(b"unit patch").hexdigest(),
                  "changed_parameters": 0, "l2": 0.0, "linf": 0.0},
        "counters": {"tool_calls": 4, "run_calls": 2, "inspect_calls": 0, "forward_tokens": 77},
    }


def write_result(root, directory, data, wrapper=None):
    path = root / directory / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    if wrapper is not None:
        (path.parent / "wrapper.json").write_text(json.dumps(wrapper))
    return path


def write_trace(root, data, *, episode_ok=True, trace_ok=True, has_reward=True, patch_sha=None):
    trace = {"id": "unit-trace", "ok": trace_ok,"is_completed":True,"errors":[],
             "rewards": {"editing_reward": {"score":data["metrics"]["reward"],"weight":1.0}} if has_reward else {},
             "agent": {"config": {"model": "unit-editor"}},
             "info": {"iwibench": {"session_id": Path(data["patch"]["path"]).parent.name,
                                    "patch": {"sha256": patch_sha or data["patch"]["sha256"]}}}}
    path = root / "traces" / "traces.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"ok": episode_ok, "traces": [trace]}) + "\n")


def test_completed_prime_trace_preserves_zero_reward_and_raw_denominators(tmp_path):
    data = result()
    write_result(tmp_path, "result", data)
    write_trace(tmp_path, data)
    rows = list(report.rows(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["run_type"] == "prime"
    assert row["editor_model"] == "unit-editor"
    assert row["reward"] == 0.0
    assert row["changed_parameters"] == 0
    assert (row["target_base_correct"], row["target_edited_correct"], row["target_count"]) == (1, 1, 3)
    assert (row["controls_retained"], row["control_count"]) == (19, 20)
    assert row["override_accuracy"] is None


@pytest.mark.parametrize("episode_ok,trace_ok,has_reward", [(False, True, True), (True, False, True), (True, True, False)])
def test_incomplete_prime_trace_cannot_certify_a_result(tmp_path, episode_ok, trace_ok, has_reward):
    data = result()
    write_result(tmp_path, "result", data)
    write_trace(tmp_path, data, episode_ok=episode_ok, trace_ok=trace_ok, has_reward=has_reward)
    with pytest.raises(ValueError, match="completed trace"):
        list(report.rows(tmp_path))


def test_completed_trace_must_match_the_terminal_patch(tmp_path):
    data = result()
    write_result(tmp_path, "result", data)
    write_trace(tmp_path, data, patch_sha="0" * 64)
    with pytest.raises(ValueError, match="patch"):
        list(report.rows(tmp_path))


@pytest.mark.parametrize("defect",["null_reward","numeric_reward","incomplete","errors","different_reward"])
def test_invalid_scoring_trace_is_rejected(tmp_path,defect):
    data = result()
    write_result(tmp_path,"result",data)
    write_trace(tmp_path,data)
    path = tmp_path/"traces"/"traces.jsonl"
    episode = json.loads(path.read_text())
    trace = episode["traces"][0]
    if defect == "null_reward":trace["rewards"]["editing_reward"] = None
    elif defect == "numeric_reward":trace["rewards"]["editing_reward"] = 0.0
    elif defect == "incomplete":trace["is_completed"] = False
    elif defect == "errors":trace["errors"] = [{"type":"ExecutionError"}]
    else:trace["rewards"]["editing_reward"]["score"] = 0.5
    path.write_text(json.dumps(episode)+"\n")
    with pytest.raises(ValueError,match="(?i)(completed trace|reward)"):
        list(report.rows(tmp_path))


def test_identical_session_copies_are_counted_once(tmp_path):
    data = result()
    write_result(tmp_path, "copy-a", data)
    write_result(tmp_path, "copy-b", copy.deepcopy(data))
    write_trace(tmp_path, data)
    assert len(list(report.rows(tmp_path))) == 1


def test_later_copy_wrapper_still_identifies_legacy_development_run(tmp_path):
    data = result(kind="unclassified")
    write_result(tmp_path, "a-service-copy", data)
    write_result(tmp_path, "z-development-package", data,
                 {"type": "development_baseline", "patch": {"sha256": data["patch"]["sha256"]}})
    rows = list(report.rows(tmp_path))
    assert len(rows) == 1
    assert rows[0]["run_type"] == "development_baseline"
    assert rows[0]["editor_model"] == ""


def test_development_label_cannot_be_silently_promoted_to_prime(tmp_path):
    data = result(kind="development_baseline")
    write_result(tmp_path, "result", data)
    write_trace(tmp_path, data)
    with pytest.raises(ValueError, match="(?i)(development|conflict|run.type)"):
        list(report.rows(tmp_path))


def test_conflicting_patch_copies_of_one_session_are_rejected(tmp_path):
    first = result(kind="development_baseline")
    second = copy.deepcopy(first)
    second["patch"]["sha256"] = "f" * 64
    write_result(tmp_path, "copy-a", first)
    write_result(tmp_path, "copy-b", second)
    with pytest.raises(ValueError, match="(?i)(patch|conflict)"):
        list(report.rows(tmp_path))


def test_cli_writes_separate_prime_and_development_tables(tmp_path):
    source = tmp_path / "runs"
    prime, development = result(session="prime-session"), result(session="development-session", kind="development_baseline")
    write_result(source, "prime", prime)
    write_result(source, "development", development)
    write_trace(source, prime)
    output = tmp_path / "tables"
    completed = subprocess.run([sys.executable, str(REPORT_PATH), str(source), "--output-dir", str(output)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    files = sorted(path.name for path in output.glob("*.csv"))
    assert files == ["gemma-3-1b-it-development_baseline.csv", "gemma-3-1b-it-prime.csv"]
    for name in files:
        with (output / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["run_type"] == ("prime" if name.endswith("-prime.csv") else "development_baseline")
        assert rows[0]["reward"] == "0.0"
