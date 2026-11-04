"""Fresh-process checkpoint-plus-patch inference for terminal grading.

Run ``python -m iwibench.grading_worker request.json`` or pass JSON on stdin.
The trusted caller owns test inputs and scoring; this worker only generates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runtime import NativeTarget


def run(request: dict) -> dict:
    allowed = {"model_id", "revision", "device", "cache_dir", "patch_path",
               "prompts", "max_new_tokens", "seeds", "patch_sha256"}
    if set(request) - allowed:
        raise ValueError("unrecognised grading worker options")
    if request.get("patch_path") and not request.get("patch_sha256"):
        raise ValueError("patch_sha256 is required with patch_path")
    target = NativeTarget(request["model_id"], request["revision"],
                          device=request.get("device", "cuda"),
                          cache_dir=request.get("cache_dir"))
    metadata = target.apply_patch(request["patch_path"], expected_sha256=request["patch_sha256"]) if request.get("patch_path") else None
    prompts, seeds = request["prompts"], request["seeds"]
    if not prompts or len(prompts) != len(seeds):
        raise ValueError("one seed is required for each nonempty grading prompt batch")
    generations = []
    for start in range(0, len(prompts), 8):
        generations.extend(target.generate(prompts[start:start + 8], request["max_new_tokens"],
                                           seeds[start:start + 8]))
    return {"model_id": target.model_id, "revision": target.revision,
            "patch_metadata": metadata, "generations": generations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", help="JSON request path; defaults to stdin")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text() if args.request else sys.stdin.read())
    json.dump(run(request), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
