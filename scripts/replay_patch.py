#!/usr/bin/env python3
"""Replay an immutable submitted patch on unchanged targets and resampled controls."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from iwibench.config import MODELS
from iwibench.families import _general_controls
from iwibench.scoring import score_episode


GROUPS = ("calibration", "targets", "nearby_controls", "general_controls", "override_controls")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay_episode(original, control_seed):
    if type(control_seed) is not int or control_seed < 0:
        raise ValueError("control_seed must be a nonnegative integer")
    ep = copy.deepcopy(original)
    ep["general_controls"] = _general_controls(control_seed, original["split"])
    old_prompts = {item["prompt"] for item in original["general_controls"]}
    new_prompts = {item["prompt"] for item in ep["general_controls"]}
    if old_prompts == new_prompts:
        raise ValueError("control seed recreates the original general controls")
    return ep, {"seed": control_seed, "count": len(ep["general_controls"]),
                "new_prompt_count": len(new_prompts - old_prompts),
                "repeated_prompt_count": len(new_prompts & old_prompts)}


def prepare(result_path, control_seed, cache_dir=None):
    result_path = Path(result_path).resolve()
    original = json.loads(result_path.read_text())
    patch = copy.deepcopy(original["patch"])
    original_ep = original["episode"]
    model, revision = original_ep["model_id"], original_ep["checkpoint_revision"]
    if MODELS.get(model) != revision or patch.get("model_id") != model or patch.get("revision") != revision:
        raise ValueError("result and patch must identify the same pinned checkpoint")
    recorded_patch = Path(patch["path"])
    adjacent = result_path.parent / recorded_patch.name
    patch_path = (adjacent if adjacent.exists() else recorded_patch).resolve()
    if sha256(patch_path) != patch["sha256"]:
        raise ValueError("submitted patch SHA256 mismatch")
    patch["path"] = str(patch_path)
    ep, controls = replay_episode(original_ep, control_seed)
    tokenization = {}
    calibration_path = None
    if ep["family"] == "alias_acquisition":
        calibration_path = next((folder / "calibration.json" for folder in (result_path.parent, patch_path.parent)
                                 if (folder / "calibration.json").exists()), None)
        if calibration_path is None:
            raise ValueError("alias replay requires the original calibration.json tokenisation record")
        calibration = json.loads(calibration_path.read_text())
        if calibration.get("episode") != original_ep:
            raise ValueError("calibration record does not match the original episode")
        tokenization = calibration.get("tokenization", {})
        if not tokenization:
            raise ValueError("original native alias tokenisation is missing")
    if cache_dir is None:
        old_request = next((folder / "grade-base.json" for folder in (result_path.parent, patch_path.parent)
                            if (folder / "grade-base.json").exists()), None)
        if old_request:
            cache_dir = json.loads(old_request.read_text()).get("cache_dir")
    items = [item for group in GROUPS for item in ep[group]]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("replay item identifiers must be unique")
    test_seed = original_ep.get("test_seed", original_ep.get("seeds", {}).get("test"))
    if type(test_seed) is not int or not 0 <= test_seed <= 2**64 - len(items):
        raise ValueError("original generation seed is missing or outside the supported range")
    base = {"model_id": model, "revision": revision, "device": "cuda", "cache_dir": cache_dir,
            "prompts": [item["prompt"] for item in items], "max_new_tokens": 256,
            "seeds": [test_seed + i for i in range(len(items))], "patch_path": None}
    edited = {**base, "patch_path": str(patch_path), "patch_sha256": patch["sha256"]}
    root = Path(__file__).resolve().parents[1]
    source_names = ["scripts/replay_patch.py", "src/iwibench/runtime.py", "src/iwibench/grading_worker.py",
                    "src/iwibench/families.py", "src/iwibench/scoring.py", "src/iwibench/config.py"]
    record = {"run_type": "replay", "status": "prepared", "parent_trace_id": original.get("trace_id", ""),
              "parent_run_type": original.get("run_type", "unclassified"),
              "parent_result_path": str(result_path), "parent_result_sha256": sha256(result_path),
              "patch": patch, "episode": ep, "general_control_realisation": controls,
              "same_target_items": True, "new_target_semantics": False,
              "tokenization": tokenization, "requests": {"base": base, "edited": edited},
              "source_sha256": {name: sha256(root / name) for name in source_names},
              "outputs": {}, "worker_seconds": {}}
    if calibration_path:
        record["original_calibration_sha256"] = sha256(calibration_path)
    return record


def validate_output(output, request):
    if not isinstance(output, dict) or output.get("model_id") != request["model_id"] or output.get("revision") != request["revision"]:
        raise ValueError("worker output checkpoint does not match the request")
    metadata = output.get("patch_metadata")
    if request["patch_path"]:
        if not isinstance(metadata, dict) or metadata.get("model_id") != request["model_id"] or metadata.get("revision") != request["revision"]:
            raise ValueError("worker output patch identity does not match the request")
    elif metadata is not None:
        raise ValueError("pristine worker unexpectedly applied a patch")
    generations = output.get("generations")
    if not isinstance(generations, list) or len(generations) != len(request["prompts"]):
        raise ValueError("worker output has incomplete sequence coverage")
    for item in generations:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("worker completion must contain text")
        if any(type(item.get(key)) is not int or item[key] < 0 for key in ("input_tokens", "output_tokens")):
            raise ValueError("worker token counts must be nonnegative integers")
        if item["output_tokens"] > request["max_new_tokens"]:
            raise ValueError("worker exceeded the fixed generation cap")


def write_record(directory, record):
    temporary = directory / "replay.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "replay.json")


def replay(result_path, control_seed, output_dir, cache_dir=None):
    directory = Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError("replay output directory already exists")
    record = prepare(result_path, control_seed, cache_dir)
    directory.mkdir(parents=True, exist_ok=False)
    write_record(directory, record)
    try:
        for phase in ("base", "edited"):
            request = record["requests"][phase]
            request_path = directory / (phase + ".request.json")
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            started = time.perf_counter()
            completed = subprocess.run([sys.executable, "-m", "iwibench.grading_worker", str(request_path)],
                                       capture_output=True, text=True, check=False)
            record["worker_seconds"][phase] = time.perf_counter() - started
            (directory / (phase + ".stdout")).write_text(completed.stdout)
            (directory / (phase + ".stderr")).write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"{phase} worker failed with exit code {completed.returncode}; see {phase}.stderr")
            output = json.loads(completed.stdout)
            validate_output(output, request)
            record["outputs"][phase] = output
            write_record(directory, record)
        if sha256(record["patch"]["path"]) != record["patch"]["sha256"]:
            raise ValueError("submitted patch changed during replay")
        items = [item for group in GROUPS for item in record["episode"][group]]
        responses = {phase: {item["id"]: generation["text"] for item, generation in
                            zip(items, record["outputs"][phase]["generations"], strict=True)}
                     for phase in ("base", "edited")}
        record["metrics"] = score_episode(record["episode"], responses["base"], responses["edited"],
                                          tokenization=record["tokenization"])
        generations = [item for output in record["outputs"].values() for item in output["generations"]]
        record["resources"] = {"sequence_evaluations": len(generations),
                               "forward_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in generations)}
        record["status"] = "complete"
        write_record(directory, record)
        return record
    except Exception as exc:
        record.pop("metrics", None)
        record["status"] = "execution_failed"
        record["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_record(directory, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="existing results.json for the submitted patch")
    parser.add_argument("--control-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", help="checkpoint cache; defaults to the original grading request's cache")
    args = parser.parse_args()
    result = replay(args.result, args.control_seed, args.output_dir, args.cache_dir)
    print(json.dumps({"run_type": "replay", "parent_trace_id": result["parent_trace_id"],
                      "same_target_items": True, "new_target_semantics": False,
                      "general_control_realisation": result["general_control_realisation"],
                      "metrics": result["metrics"], "output": str(args.output_dir / "replay.json")}))


if __name__ == "__main__":
    main()
