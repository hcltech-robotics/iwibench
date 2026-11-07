#!/usr/bin/env python3
"""Report editor usage for the completed episodes selected in result CSVs."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = [
    ("Terra 6.1", "results/ten-families-v022/gemma-3-4b-it-prime.csv", "runs/terra-ten-families-v022/attempts"),
    ("Sol 6.1", "results/ten-families-v022-sol/gemma-3-4b-it-prime.csv", "runs/sol-ten-families-v023-retry1/attempts"),
    ("Astra", "results/ten-families-v022-astra/gemma-3-4b-it-prime.csv", "runs/astra-ten-families/attempts"),
    ("GLM 5.3", "results/ten-families-v022-glm/gemma-3-4b-it-prime.csv", "runs/glm-ten-families-complete/attempts"),
    ("DeepSeek V4.1 Flash", "results/ten-families-v022-deepseek/gemma-3-4b-it-prime.csv", "runs/deepseek-ten-families/attempts"),
    ("Nemotron 3 Ultra", "results/ten-families-v022-nemotron/gemma-3-4b-it-prime.csv", "runs/nemotron-ten-families/attempts"),
]
FAMILIES = {"alias_acquisition", "output_code_remapping", "lexical_inhibition",
            "terminology_preference", "default_response_language", "structured_response_default",
            "answer_length_control", "factual_update", "selective_factual_withholding",
            "resistance_to_false_suggestions"}


def call_usage(usage):
    """Convert Verifiers v1 split-cache usage into inclusive input and output."""
    if not isinstance(usage, dict):
        raise ValueError("Missing call usage")
    for key in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError(f"Missing or invalid {key}")
    for key in ("cached_input_tokens", "reasoning_tokens"):
        if key in usage and usage[key] is not None and (type(usage[key]) is not int or usage[key] < 0):
            raise ValueError(f"Invalid {key}")
    cached = usage.get("cached_input_tokens")
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None and reasoning > usage["completion_tokens"]:
        reasoning = None
    return {"input_tokens": usage["prompt_tokens"] + (cached or 0),
            "output_tokens": usage["completion_tokens"],
            "cached_input_tokens": cached, "reasoning_tokens": reasoning}


def sum_known(values):
    return sum(values) if all(value is not None for value in values) else None


def validate_rows(rows):
    if len(rows) != 30 or len({r["trace_id"] for r in rows}) != 30:
        raise ValueError("Expected 30 distinct selected episodes")
    if Counter(r["family"] for r in rows) != Counter({family: 3 for family in FAMILIES}):
        raise ValueError("Expected exactly three episodes in each of ten families")
    for field in ("editor_model", "model_id"):
        if len({r[field] for r in rows}) != 1 or not rows[0][field]:
            raise ValueError(f"Expected one consistent {field}")


def report(label, csv_path, attempts):
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    validate_rows(rows)
    traces = {}
    for path in attempts.glob("*/traces/*/traces.jsonl"):
        with path.open(encoding="utf-8") as source:
            for line in source:
                for trace in json.loads(line)["traces"]:
                    if trace["id"] in traces:
                        raise ValueError(f"Duplicate trace ID {trace['id']}")
                    traces[trace["id"]] = trace, path
    episodes = []
    for row in rows:
        trace_id = row["trace_id"]
        if trace_id not in traces:
            raise ValueError(f"Missing trace {trace_id}")
        trace, path = traces[trace_id]
        calls = trace["calls"]
        if not calls:
            raise ValueError(f"No calls for {trace_id}")
        for call in calls:
            if call.get("model") is not None and call["model"] != row["editor_model"]:
                raise ValueError(f"Call model mismatch for {trace_id}")
        # Failed provider requests can recover within a scored episode. Preserve
        # their call count, but do not invent zero usage for an unreported call.
        unreported = [{"call_index": i, "error": call["error"]}
                      for i, call in enumerate(calls)
                      if call.get("usage") is None
                      and isinstance(call.get("error"), dict)
                      and call["error"].get("type") == "ProviderError"]
        unreported_indices = {item["call_index"] for item in unreported}
        reported_calls = [(i, call) for i, call in enumerate(calls)
                          if i not in unreported_indices]
        if not reported_calls:
            raise ValueError(f"No reported usage for {trace_id}")
        usage = [call_usage(call.get("usage")) for _, call in reported_calls]
        conflicts = [{"call_index": i, "completion_tokens": call["usage"]["completion_tokens"],
                      "reported_reasoning_tokens": call["usage"]["reasoning_tokens"]}
                     for i, call in reported_calls
                     if call["usage"].get("reasoning_tokens") is not None
                     and call["usage"]["reasoning_tokens"] > call["usage"]["completion_tokens"]]
        inp = sum(item["input_tokens"] for item in usage)
        out = sum(item["output_tokens"] for item in usage)
        reward = float(row["reward"])
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError(f"Invalid reward for {trace_id}")
        target_forward_tokens = int(row["forward_tokens"])
        if target_forward_tokens < 0:
            raise ValueError(f"Invalid target forward tokens for {trace_id}")
        episodes.append({"trace_id": trace_id, "family": row["family"],
                         "parameter_seed": int(row["parameter_seed"]), "reward": reward,
                         "trace_path": str(path.relative_to(ROOT)), "calls": len(calls),
                         "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                         "target_forward_tokens": target_forward_tokens,
                         "cached_input_tokens": sum_known([u["cached_input_tokens"] for u in usage]),
                         "reasoning_tokens": sum_known([u["reasoning_tokens"] for u in usage])})
        if unreported:
            episodes[-1]["failed_calls_without_reported_usage"] = unreported
        if conflicts:
            episodes[-1]["reasoning_usage_conflicts"] = conflicts
    total = sum(e["total_tokens"] for e in episodes)
    if total == 0:
        raise ValueError("Editor token denominator is zero")
    reward_sum = sum(e["reward"] for e in episodes)
    result = {"editor": label, "results_csv": str(csv_path.relative_to(ROOT)),
            "editor_model": rows[0]["editor_model"], "target_model_id": rows[0]["model_id"],
            "episodes": episodes, "episode_count": len(episodes),
            "calls": sum(e["calls"] for e in episodes),
            "target_forward_tokens": sum(e["target_forward_tokens"] for e in episodes),
            "sum_reward": reward_sum, "macro_reward": reward_sum / len(episodes),
            "input_tokens": sum(e["input_tokens"] for e in episodes),
            "output_tokens": sum(e["output_tokens"] for e in episodes),
            "total_tokens": total,
            "cached_input_tokens": sum_known([e["cached_input_tokens"] for e in episodes]),
            "reasoning_tokens": sum_known([e["reasoning_tokens"] for e in episodes]),
            "mean_tokens_per_episode": total / len(episodes),
            "reward_per_1000_tokens": 1000 * reward_sum / total}
    missing_count = sum(len(e.get("failed_calls_without_reported_usage", [])) for e in episodes)
    if missing_count:
        result["failed_calls_without_reported_usage"] = missing_count
        result["usage_caveat"] = "Totals include only reported usage. Failed provider calls without usage remain in call counts and may have unreported charges."
    conflict_count = sum(len(e.get("reasoning_usage_conflicts", [])) for e in episodes)
    if conflict_count:
        result["reasoning_usage_conflict_count"] = conflict_count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")
    parser.add_argument("--editor", action="append", nargs=3, metavar=("LABEL", "RESULTS_CSV", "ATTEMPTS"),
                        help="Editor label, results CSV and attempts directory. Repeat for each editor.")
    args = parser.parse_args()
    editors = args.editor if args.editor is not None else DEFAULTS
    output = {"scope": "Selected completed episodes only. Other attempts and retries are excluded.",
              "accounting": "Totals reported by the provider. Verifiers v1 prompt_tokens excludes recorded cached reads. Input includes them. Reasoning is included in output. Null means a missing or inconsistent breakdown.",
              "verifiers_commit": "5dad16d40e2d025672fdb7aa75b6540c0053e4e0",
              "editors": [report(name, ROOT / csv_path, ROOT / attempts) for name, csv_path, attempts in editors]}
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
