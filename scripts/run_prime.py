#!/usr/bin/env python3
"""Run local Verifiers traces against an existing IWIBench GPU service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib


def load_credentials(path: Path, environment: dict[str, str]) -> None:
    allowed = {"OPENAI_API_KEY", "IWIBENCH_SERVICE_URL", "IWIBENCH_ADMIN_TOKEN", "IWIBENCH_EDITOR_TOKEN"}
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip().startswith("export "):
            line = line.strip()[7:]
        key, equals, value = line.partition("=")
        key = key.strip()
        if not equals or key not in allowed or environment.get(key):
            continue
        fields = shlex.split(value, comments=True)
        if len(fields) != 1:
            raise ValueError(f"Invalid value for {key} in {path}")
        environment[key] = fields[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    from iwibench.families import AVAILABLE_FAMILIES
    parser.add_argument("--family", choices=AVAILABLE_FAMILIES, required=True)
    parser.add_argument("--model", choices=["google/gemma-3-1b-it", "google/gemma-3-4b-it"], default="google/gemma-3-1b-it")
    parser.add_argument("--editor-model", default="gpt-6.1-terra")
    parser.add_argument("--editor-base-url", default="http://127.0.0.1:8770/v1")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Override the editor's reasoning setting; preserves the existing default when omitted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".env")
    parser.add_argument("--service-env", type=Path)
    parser.add_argument("--service-url", default="http://127.0.0.1:8768")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/prime"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Start a separate process and print its PID and log path")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or args.num_seeds < 1:
        parser.error("seed must be nonnegative and num-seeds must be positive")
    reasoning_model = args.editor_model.startswith(("gpt-5", "gpt-6"))
    sol = args.editor_model == "gpt-6.1-sol"
    if sol and args.reasoning_effort in {"none", "minimal"}:
        parser.error("gpt-6.1-sol requires reasoning effort low, medium, high or xhigh")
    if reasoning_model and args.editor_base_url.rstrip("/") == "https://api.openai.com/v1":
        parser.error("Use --editor-base-url http://127.0.0.1:8770/v1 for GPT-5/GPT-6 completion-token transport.")
    root = Path(__file__).resolve().parents[1]
    if args.detach:
        log_path = (args.log_file or args.output_dir / f"runner-{args.family}-{args.seed}-{time.time_ns()}.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *[arg for arg in sys.argv[1:] if arg != "--detach"]]
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(log_path)}))
        return 0
    environment = os.environ.copy()
    load_credentials(args.env_file, environment)
    if args.service_env:
        load_credentials(args.service_env, environment)
    environment["IWIBENCH_SERVICE_URL"] = args.service_url
    if not args.dry_run:
        missing = [name for name in ["OPENAI_API_KEY", "IWIBENCH_ADMIN_TOKEN", "IWIBENCH_EDITOR_TOKEN"] if not environment.get(name)]
        if missing:
            parser.error("Missing credentials: " + ", ".join(missing))
    with (root / "configs/prime.toml").open("rb") as source:
        config = tomllib.load(source)
    config["model"] = args.editor_model
    config["client"] = {"base_url": args.editor_base_url, "api_key_var": "OPENAI_API_KEY"}
    config["output_dir"] = str(args.output_dir.resolve())
    config["env"]["taskset"].update(
        families=[args.family], model_id=args.model, seed=args.seed, num_seeds=args.num_seeds,
    )
    config["env"]["agent"].update(harness={"id": "null"}, max_turns=64)
    config["sampling"].update(temperature=None if args.editor_model.startswith("gpt-5") or sol else 0,
                              reasoning_effort=args.reasoning_effort if args.reasoning_effort is not None else
                              ("low" if sol else "none" if reasoning_model else None),
                              max_tokens=4096)
    with tempfile.TemporaryDirectory(prefix="iwibench-prime-") as directory:
        path = Path(directory) / "run.json"
        path.write_text(json.dumps(config))
        command = [str(Path(sys.executable).with_name("vf-run")), "iwibench", "@", str(path)]
        if args.dry_run:
            command.append("--dry-run")
        # vf-run stores traces locally and has no publishing step or --no-push flag.
        return subprocess.run(command, cwd=root, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
