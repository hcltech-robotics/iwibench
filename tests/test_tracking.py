"""Mocked W&B retry tests. These make no live publication claims."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from iwibench.tracking import log_result


@pytest.fixture
def measured_result(tmp_path):
    directory = tmp_path / "unit-session-0123456789abcdef"
    directory.mkdir()
    patch = directory / "patch.safetensors"
    patch.write_bytes(b"unit-test patch bytes, not a model")
    return {
        "trace_id": "unit-trace", "run_type": "development_baseline",
        "patch": {"path": str(patch), "model_id": "unit-target", "revision": "a" * 40,
                  "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                  "changed_parameters": 0, "l2": 0.0, "linf": 0.0},
        "metrics": {"family": "alias_acquisition", "episode_id": "unit-episode", "reward": 0.0,
                    "controls_retained": 19, "control_retention_denominator": 20,
                    "control_retention": 0.95},
        "counters": {"forward_tokens": 77, "tool_calls": 4},
    }


class FakeWandb:
    """A call recorder with a single injected failure; never performs I/O."""
    def __init__(self, fail_at=None, fail_finish=False):
        self.fail_at = fail_at
        self.fail_finish = fail_finish
        self.init_calls = []
        self.logs = []
        self.artifacts = []
        self.finish_calls = []

    def maybe_fail(self, stage):
        if self.fail_at == stage:
            self.fail_at = None
            raise RuntimeError("planted " + stage + " failure")

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.maybe_fail("init")
        parent = self

        class Run:
            id = kwargs["id"]
            url = "https://unit.invalid/" + id

            def log(self, values):
                parent.logs.append(copy.deepcopy(values))
                parent.maybe_fail("log")

            def log_artifact(self, artifact):
                parent.artifacts.append(artifact)
                parent.maybe_fail("log_artifact")

            def finish(self, **options):
                parent.finish_calls.append(options)
                if parent.fail_finish:
                    raise RuntimeError("planted finish failure")
                parent.maybe_fail("finish")

        return Run()

    def Artifact(self, name, type, metadata):
        artifact = SimpleNamespace(name=name, type=type, metadata=copy.deepcopy(metadata), files=[])
        artifact.add_file = lambda path: artifact.files.append(path)
        return artifact


def test_success_records_zero_values_and_repeated_call_does_not_publish_again(tmp_path, monkeypatch, measured_result):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    receipt_path = tmp_path / "receipt.json"
    first = log_result(measured_result, "unit-project", receipt_path)
    second = log_result(measured_result, "unit-project", receipt_path)
    assert first == second == json.loads(receipt_path.read_text())
    assert len(fake.init_calls) == 1
    assert fake.init_calls[0]["config"]["run_type"] == "development_baseline"
    assert fake.logs[0]["reward"] == 0.0
    assert fake.logs[0]["patch/changed_parameters"] == 0
    assert fake.logs[0]["controls_retained"] == 19
    assert fake.logs[0]["control_retention_denominator"] == 20
    assert fake.logs[0]["forward_tokens"] == 77
    assert fake.logs[0]["patch/bytes"] == Path(measured_result["patch"]["path"]).stat().st_size
    assert fake.artifacts[0].metadata["sha256"] == measured_result["patch"]["sha256"]


@pytest.mark.parametrize("stage", ["init", "log", "log_artifact", "finish"])
def test_failure_leaves_no_success_receipt_and_retry_reuses_run_id(tmp_path, monkeypatch, measured_result, stage):
    fake = FakeWandb(fail_at=stage)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="planted " + stage):
        log_result(measured_result, "unit-project", receipt_path)
    assert not receipt_path.exists()
    receipt = log_result(measured_result, "unit-project", receipt_path)
    assert receipt["trace_id"] == "unit-trace"
    assert fake.init_calls[0]["id"] == fake.init_calls[1]["id"]
    assert all(call["resume"] == "allow" for call in fake.init_calls)


@pytest.mark.parametrize("change", ["patch", "trace", "project"])
def test_receipt_cannot_be_reused_for_a_different_publication(tmp_path, monkeypatch, measured_result, change):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    receipt_path = tmp_path / "receipt.json"
    log_result(measured_result, "unit-project", receipt_path)
    other = copy.deepcopy(measured_result)
    project = "unit-project"
    if change == "patch":
        other["patch"]["sha256"] = "f" * 64
    elif change == "trace":
        other["trace_id"] = "different-trace"
    else:
        project = "other-project"
    with pytest.raises(ValueError, match="(?i)(receipt|identity|publication|mismatch)"):
        log_result(other, project, receipt_path)
    assert len(fake.init_calls) == 1


def test_cleanup_failure_does_not_hide_the_upload_failure(tmp_path, monkeypatch, measured_result):
    fake = FakeWandb(fail_at="log_artifact", fail_finish=True)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with pytest.raises(RuntimeError, match="planted log_artifact failure"):
        log_result(measured_result, "unit-project", tmp_path / "receipt.json")


def test_receipt_write_failure_cannot_leave_a_partial_success_receipt(tmp_path, monkeypatch, measured_result):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    receipt_path = tmp_path / "receipt.json"
    original_write = Path.write_text

    def partial_write(path, text, *args, **kwargs):
        original_write(path, text[:10], *args, **kwargs)
        raise OSError("planted disk write failure")

    monkeypatch.setattr(Path, "write_text", partial_write)
    with pytest.raises(OSError, match="planted disk"):
        log_result(measured_result, "unit-project", receipt_path)
    assert not receipt_path.exists()
