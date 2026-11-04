#!/usr/bin/env python3
"""Export measured terminal results into one CSV per target checkpoint."""
import argparse
import csv
import json
import math
from pathlib import Path


def rows(root):
    traces = {}
    for path in sorted(root.rglob("traces.jsonl")):
        for line in path.read_text().splitlines():
            episode = json.loads(line)
            for trace in episode.get("traces",[]):
                info = trace.get("info",{}).get("iwibench",{})
                reward = trace.get("rewards",{}).get("editing_reward")
                score = reward.get("score") if isinstance(reward,dict) else None
                valid_score = isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and 0 <= score <= 1
                if episode.get("ok") and trace.get("ok") and trace.get("is_completed") and not trace.get("errors") and valid_score:
                    traces[info["session_id"]] = trace
    sessions = {}
    for path in sorted(root.rglob("results.json")):
        data = json.loads(path.read_text())
        if not {"episode","metrics","patch","counters"} <= data.keys():
            continue
        session_id = Path(data["patch"]["path"]).parent.name
        sessions.setdefault(session_id,[]).append((path,data))
    for session_id,copies in sessions.items():
        if len({data["patch"]["sha256"] for _,data in copies}) != 1:
            raise ValueError("Conflicting patch hashes for one session")
        if len({json.dumps(data["metrics"],sort_keys=True) for _,data in copies}) != 1:
            raise ValueError("Conflicting metrics for one session")
        path,data = copies[0]
        ep, metrics, patch, counters = (data[k] for k in ["episode","metrics","patch","counters"])
        trace = traces.get(session_id)
        kinds = {d.get("run_type","unclassified") for _,d in copies}-{"unclassified"}
        for candidate,_ in copies:
            if (candidate.parent/"wrapper.json").exists():
                wrapper = json.loads((candidate.parent/"wrapper.json").read_text())
                if wrapper.get("type") == "development_baseline" and wrapper.get("patch",{}).get("sha256") == patch["sha256"]:
                    kinds.add("development_baseline")
        if len(kinds)>1:
            raise ValueError("Conflicting run types for one session")
        kind = next(iter(kinds),"unclassified")
        trace_id = data.get("trace_id","")
        editor = ""
        if trace:
            if kind == "development_baseline":
                raise ValueError("Development baseline conflicts with a Prime trace")
            if trace["info"]["iwibench"]["patch"]["sha256"] != patch["sha256"]:
                raise ValueError("Trace patch differs from terminal patch")
            if not math.isclose(trace["rewards"]["editing_reward"]["score"],metrics["reward"],abs_tol=1e-12):
                raise ValueError("Trace reward differs from terminal reward")
            kind, trace_id = "prime", trace["id"]
            editor = trace["agent"]["config"]["model"]
        if kind == "prime" and trace is None:
            raise ValueError(f"Prime result lacks its completed trace: {path}")
        target = metrics["groups"]["targets"]
        yield {
            "run_type":kind,"editor_model":editor,
            "model_id":ep["model_id"],"revision":ep["checkpoint_revision"],"family_version":ep["family_version"],
            "family":ep["family"],"parameter_seed":ep["parameter_seed"],"trace_id":trace_id,
            "target_base_correct":target["base_successes"],"target_edited_correct":target["edited_successes"],"target_count":target["count"],
            "controls_retained":metrics["controls_retained"],"control_count":metrics["control_retention_denominator"],
            "control_retention":metrics["control_retention"],"local_spillover":metrics["local_spillover"],
            "override_accuracy":metrics["override_accuracy"],"gain":metrics["gain"],"reward":metrics["reward"],
            "tool_calls":counters["tool_calls"],"run_calls":counters["run_calls"],"inspect_calls":counters["inspect_calls"],
            "forward_tokens":counters["forward_tokens"],"changed_parameters":patch["changed_parameters"],
            "patch_l2":patch["l2"],"patch_linf":patch["linf"],"patch_sha256":patch["sha256"],"source":str(path),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--editor",help="Restrict Prime tables to this exact editor model")
    args = parser.parse_args()
    grouped = {}
    for row in rows(args.runs):
        if args.editor and row["run_type"] == "prime" and row["editor_model"] != args.editor:
            continue
        grouped.setdefault((row["model_id"],row["run_type"]),[]).append(row)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for (model,kind),records in grouped.items():
        path = args.output_dir/(model.split("/")[-1]+"-"+kind+".csv")
        with path.open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({"model":model,"attempts":len(records),"path":str(path)}))


if __name__ == "__main__":
    main()
