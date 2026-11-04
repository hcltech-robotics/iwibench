"""Measure development capability and editing headroom on pristine native weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from iwibench.config import MODELS
from iwibench.families import AVAILABLE_FAMILIES, FAMILY_VERSION, episode
from iwibench.scoring import calibrate_episode, check_item


def make_record(ep: dict, generations: list[dict], tokenization: dict,
                elapsed_seconds: float) -> dict:
    """Join measured completions to their exact inputs, without dropping failures."""
    items = ep["calibration"] + ep["targets"]
    if len(generations) != len(items):
        raise ValueError("one generation is required for every calibration and target item")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("calibration and target item identifiers must be unique")
    outputs = {item["id"]: generation["text"] for item, generation in zip(items, generations, strict=True)}
    calibration = calibrate_episode(ep, outputs, tokenization)
    target_successes = sum(check_item(item, outputs[item["id"]])["success"] for item in ep["targets"])
    target_count = len(ep["targets"])
    if not target_count:
        raise ValueError("at least one baseline target item is required")
    headroom = target_successes < target_count
    records = []
    for item, generation in zip(items, generations, strict=True):
        if any(type(generation[key]) is not int or generation[key] < 0
               for key in ("input_tokens", "output_tokens")):
            raise ValueError("generation token counts must be nonnegative integers")
        records.append({"item": item, **generation, "checks": check_item(item, generation["text"])})
    return {
        "episode": ep, "outputs": outputs, "generations": records,
        "tokenization": tokenization, "calibration": calibration,
        "target_successes": target_successes, "target_count": target_count,
        "target_success_base": target_successes / target_count,
        "baseline_headroom": headroom,
        "eligible_for_edit_attempt": calibration["eligible"] and headroom,
        "sequence_evaluations": len(generations),
        "input_tokens": sum(g["input_tokens"] for g in generations),
        "output_tokens": sum(g["output_tokens"] for g in generations),
        "forward_tokens": sum(g["input_tokens"] + g["output_tokens"] for g in generations),
        "elapsed_seconds": elapsed_seconds,
    }


def save_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="google/gemma-3-1b-it")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eligible-per-family", type=int,
                        help="Keep the first N eligible, distinct parameter sets per family")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--families", nargs="+", choices=AVAILABLE_FAMILIES, default=list(AVAILABLE_FAMILIES))
    args = parser.parse_args()
    if args.seeds < 1 or args.max_new_tokens < 1:
        parser.error("seeds and max-new-tokens must be positive")
    if args.eligible_per_family is not None and not 1 <= args.eligible_per_family <= args.seeds:
        parser.error("eligible-per-family must be positive and no greater than seeds")
    if len(set(args.families)) != len(args.families):
        parser.error("families must not repeat")
    if args.output.exists():
        parser.error("output already exists; select a new path to preserve previous measurements")
    from iwibench.runtime import NativeTarget, alias_tokenization

    target = NativeTarget(args.model, MODELS[args.model], cache_dir=args.cache_dir)
    root = Path(__file__).resolve().parents[1]
    sources = sorted(str(path.relative_to(root))
                     for path in (root / "src/iwibench").rglob("*")
                     if path.suffix in {".py", ".json"})
    sources.append("scripts/calibrate.py")
    sources.append("data/facts.json")
    result = {
        "model_id": args.model, "revision": MODELS[args.model],
        "family_version": FAMILY_VERSION, "split": "development",
        "weights": "pristine", "generation": target.manifest()["generation"],
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "huggingface-hub", "safetensors")},
        "max_new_tokens": args.max_new_tokens, "batch_size": 8,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sources if (root / name).exists()},
        "episodes": [],
        "selection": {
            "policy": "first eligible distinct parameters in ascending seed order",
            "eligible_per_family": args.eligible_per_family,
            "candidate_seed_limit": args.seeds,
            "families": args.families,
            "complete": False,
        },
    }
    save_result(args.output, result)
    for family in args.families:
        seen_parameters = set()
        eligible_count = 0
        for seed in range(args.seeds):
            ep = episode(family, seed, model_id=args.model)
            identity = json.dumps(ep["parameters"], sort_keys=True, separators=(",", ":"))
            if args.eligible_per_family is not None and identity in seen_parameters:
                continue
            seen_parameters.add(identity)
            tokenization = alias_tokenization(ep, target)
            items = ep["calibration"] + ep["targets"]
            started = time.perf_counter()
            generations = []
            for offset in range(0, len(items), 8):
                batch = items[offset:offset + 8]
                seeds = [(ep["seeds"]["probe"] + offset + i) % (2**63) for i in range(len(batch))]
                completions = target.generate([item["prompt"] for item in batch], args.max_new_tokens, seeds)
                generations.extend({**completion, "seed": generation_seed}
                                   for completion, generation_seed in zip(completions, seeds, strict=True))
            record = make_record(ep, generations, tokenization, time.perf_counter() - started)
            result["episodes"].append(record)
            save_result(args.output, result)
            print(json.dumps({"family": family, "seed": seed,
                              "eligible": record["eligible_for_edit_attempt"],
                              "calibration_correct": record["calibration"]["correct_count"],
                              "calibration_count": record["calibration"]["calibration_count"],
                              "target_successes": record["target_successes"],
                              "target_count": record["target_count"],
                              "forward_tokens": record["forward_tokens"],
                              "seconds": round(record["elapsed_seconds"], 3)}), flush=True)
            eligible_count += record["eligible_for_edit_attempt"]
            if args.eligible_per_family is not None and eligible_count == args.eligible_per_family:
                break
        if args.eligible_per_family is not None and eligible_count < args.eligible_per_family:
            raise RuntimeError(
                f"{family}: found {eligible_count} eligible distinct instances, "
                f"but {args.eligible_per_family} are required"
            )
    result["selection"]["complete"] = True
    save_result(args.output, result)


if __name__ == "__main__":
    main()
