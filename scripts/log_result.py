#!/usr/bin/env python3
"""Retry W&B logging without generating or changing a terminal model score."""
import argparse
import json
from pathlib import Path
from iwibench.tracking import log_result

parser = argparse.ArgumentParser()
parser.add_argument("result",type=Path)
parser.add_argument("--project",default="iwibench-dev")
parser.add_argument("--run-type",choices=["prime","development_baseline"])
parser.add_argument("--trace-id")
args = parser.parse_args()
result = json.loads(args.result.read_text())
if args.run_type:
    result["run_type"] = args.run_type
if args.trace_id:
    result["trace_id"] = args.trace_id
print(json.dumps(log_result(result,args.project,args.result.parent/"wandb.json")))
