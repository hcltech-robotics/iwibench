#!/usr/bin/env python3
"""Run editor attempts only for development instances that passed calibration."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
import math
from pathlib import Path
import subprocess
import sys
import time


def completed_score(directory):
    episodes = []
    try:
        for path in Path(directory).rglob("traces.jsonl"):
            episodes.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    except (OSError, json.JSONDecodeError):
        return False
    if len(episodes) != 1 or episodes[0].get("ok") is not True:
        return False
    traces = episodes[0].get("traces",[])
    if len(traces) != 1:
        return False
    trace = traces[0]
    reward = trace.get("rewards",{}).get("editing_reward",{})
    score = reward.get("score") if isinstance(reward,dict) else None
    metadata = trace.get("info",{}).get("iwibench",{})
    return (trace.get("ok") is True and trace.get("is_completed") is True and not trace.get("errors") and
            isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and
            0 <= score <= 1 and bool(metadata.get("patch",{}).get("sha256")) and
            "metrics" in metadata)


def process_identity(pid):
    """Return Linux start time for a live process; zombies have already finished."""
    if not isinstance(pid,int) or isinstance(pid,bool) or pid <= 0:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] == "Z":
        return None
    return fields[19]


def process_alive(previous):
    identity = process_identity(previous.get("pid"))
    return identity is not None and (
        previous.get("pid_start_time") is None or previous["pid_start_time"] == identity
    )


def write_status(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record)+"\n")
    os.replace(temporary,path)


def run_attempt(path, family, seed, command, poll_seconds=1):
    """Claim one directory, resume its child, or start exactly one editor attempt."""
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    status = path/"execution.json"
    traces = path/"traces"
    with (path/".execution.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous = json.loads(status.read_text()) if status.exists() else None
        if previous is not None:
            while process_alive(previous):
                time.sleep(poll_seconds)
            if completed_score(traces):
                recovered = previous.get("completed_score") is not True
                if recovered:
                    previous.update(returncode=previous.get("returncode"),completed_score=True,recovered=True)
                    write_status(status,previous)
                print(json.dumps({"family":family,"seed":seed,"status":"recovered" if recovered else "already completed"}),flush=True)
                return True
            print(json.dumps({"family":family,"seed":seed,"status":"failed",
                              "error":f"Previous attempt has no completed score; retry requires a new output directory: {path}"}),flush=True)
            return False
        # A supervisor can stop between claiming the directory and recording its
        # child PID. Existing traces/logs still forbid starting another attempt.
        if (path/"editor.log").exists() or any(traces.rglob("traces.jsonl")):
            if completed_score(traces):
                write_status(status,{"pid":None,"family":family,"seed":seed,"returncode":None,
                                     "completed_score":True,"recovered":True})
                print(json.dumps({"family":family,"seed":seed,"status":"recovered"}),flush=True)
                return True
            print(json.dumps({"family":family,"seed":seed,"status":"failed",
                              "error":f"Existing attempt requires a new retry output directory: {path}"}),flush=True)
            return False
        record = {"pid":None,"family":family,"seed":seed,"returncode":None}
        write_status(status,record)
        with (path/"editor.log").open("x") as log:
            # The child retains the lock if this supervisor exits before wait().
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            record.update(pid=process.pid,pid_start_time=process_identity(process.pid))
            write_status(status,record)
            print(json.dumps({"family":family,"seed":seed,"status":"started","pid":process.pid,"log":str(path/"editor.log")}),flush=True)
            result = process.wait()
        scored = result == 0 and completed_score(traces)
        record.update(returncode=result,completed_score=scored)
        write_status(status,record)
        print(json.dumps({"family":family,"seed":seed,"status":"finished","returncode":result,"completed_score":scored}),flush=True)
        return scored


def execute_attempts(attempts, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_attempt,*attempt):attempt for attempt in attempts}
        for future in as_completed(futures):
            path,family,seed,_ = futures[future]
            try:
                failures += not future.result()
            except Exception as error:
                failures += 1
                print(json.dumps({"family":family,"seed":seed,"status":"failed","error":str(error)}),flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration",type=Path)
    parser.add_argument("--service-env",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--families",nargs="+")
    parser.add_argument("--first-per-family",action="store_true")
    parser.add_argument("--editor-model",default="gpt-6.1-terra")
    parser.add_argument("--editor-base-url",default="http://127.0.0.1:8770/v1")
    parser.add_argument("--reasoning-effort",choices=["none","minimal","low","medium","high","xhigh"])
    parser.add_argument("--service-url",default="http://127.0.0.1:8768")
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("selection", {}).get("complete") is False:
        raise ValueError("Calibration selection is incomplete")
    from iwibench.families import FAMILY_VERSION
    if calibration["family_version"] != FAMILY_VERSION:
        raise ValueError("Calibration family version differs from installed package")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selected = set()
    attempts = []
    for record in calibration["episodes"]:
        if not record["eligible_for_edit_attempt"]:
            continue
        ep = record["episode"]
        family = ep["family"]
        if args.families and family not in args.families:
            continue
        if args.first_per_family and family in selected:
            continue
        selected.add(family)
        seed = ep["parameter_seed"] if "parameter_seed" in ep else ep["seeds"]["parameter"]
        path = args.output_dir/f"{family}-{seed}"
        command = [sys.executable,str(Path(__file__).with_name("run_prime.py")),"--family",family,
                   "--seed",str(seed),"--model",calibration["model_id"],"--service-env",str(args.service_env.resolve()),
                   "--output-dir",str(path.resolve()/"traces"),"--editor-model",args.editor_model,
                   "--editor-base-url",args.editor_base_url,"--service-url",args.service_url]
        if args.reasoning_effort is not None:
            command.extend(["--reasoning-effort",args.reasoning_effort])
        attempts.append((path,family,seed,command))
    return execute_attempts(attempts,args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
