#!/usr/bin/env python3
"""Start a detached local GPU worker using a private credential file."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default="google/gemma-3-1b-it")
    parser.add_argument("--cache-dir",required=True)
    parser.add_argument("--output-dir",default="runs/service")
    parser.add_argument("--port",type=int,default=8768)
    parser.add_argument("--wandb-project",default="iwibench-dev")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root/args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    credentials = root/".service.env"
    if not credentials.exists():
        fd = os.open(credentials,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w") as f:
            for name in ["IWIBENCH_ADMIN_TOKEN","IWIBENCH_EDITOR_TOKEN"]:
                f.write(name+"="+secrets.token_urlsafe(32)+"\n")
    environment = os.environ.copy()
    for line in credentials.read_text().splitlines():
        key,value = line.split("=",1)
        environment[key] = value
    environment["PYTHONPATH"] = str(root/"src")
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["IWIBENCH_WANDB_PROJECT"] = args.wandb_project
    environment["WANDB_DIR"] = str(output)
    with (output/"server.log").open("a") as log:
        process = subprocess.Popen([sys.executable,"-m","iwibench.service","--model",args.model,
                                    "--cache-dir",args.cache_dir,"--output-dir",str(output),"--port",str(args.port)],
                                   env=environment,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (output/"server.json").write_text(json.dumps({"pid":process.pid,"model":args.model,"port":args.port,"credential_file":str(credentials)},indent=2)+"\n")
    print(json.dumps({"pid":process.pid,"log":str(output/"server.log"),"port":args.port}))


if __name__ == "__main__":
    main()
