"""Unscored intervention recipes using IWIBench's four editor tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from iwibench.client import ServiceClient


def embedding_tensor(public: dict) -> dict:
    matches = [entry for entry in public["manifest"]["tensors"]
               if entry["writable"] and len(entry["shape"]) == 2
               and any(name.endswith("embed_tokens.weight") for name in entry["aliases"])
               and "lm_head.weight" in entry["aliases"]]
    if len(matches) != 1:
        raise ValueError("recipe requires one canonical tied input/output embedding")
    return matches[0]


def alias_edit(public: dict) -> dict:
    match = re.search(r"Make the existing word '([^']+)' refer to ([^.]+)\.", public["objective"])
    if match is None:
        raise ValueError("public objective has no supported alias binding")
    alias, referent = match.groups()
    aliases = public["token_ids"]["alias_context"]
    destination = aliases[alias]["token_ids"]
    source = public["token_ids"].get(" " + referent)
    if len(destination) != 1 or not isinstance(source, list) or len(source) != 1:
        raise ValueError("copy recipe requires single contextual alias and referent tokens")
    if source == destination:
        raise ValueError("alias copy must change a distinct token row")
    tensor = embedding_tensor(public)
    width = tensor["shape"][1]
    return {"tensor": tensor["name"], "operation": "copy",
            "selection": [destination[0], [0, width]],
            "source": {"tensor": tensor["name"], "selection": [source[0], [0, width]]}}


def lexical_forms(public: dict) -> list[str]:
    match = re.search(r"listed forms in generated answers: ([^.]+)\.", public["objective"])
    if match is None:
        raise ValueError("public objective has no supported inhibited-word list")
    return list(dict.fromkeys(word.strip() for word in match[1].split(",") if word.strip()))


def lexical_rows(public: dict, inspected_forms: dict[str, dict]) -> tuple[list[int], dict]:
    """Use complete lexical tokens only; never suppress a subword component."""
    forms = lexical_forms(public)
    selected = {}
    skipped = {}
    for word in forms:
        for spelling in (word, " " + word):
            ids = public["token_ids"].get(spelling)
            if isinstance(ids, list) and len(ids) == 1:
                selected[spelling] = ids[0]
            else:
                skipped[spelling] = {"reason": "not one disclosed token", "token_ids": ids}
    for spelling, inspection in inspected_forms.items():
        # Native Gemma prepends exactly one BOS. Position 1 identifies a whole
        # supplied word only when the complete teacher-forced input has 2 tokens.
        ids = inspection["selected_token_ids"]
        if inspection["input_tokens"] == 2 and len(ids) == 1:
            selected[spelling] = ids[0]
        else:
            skipped[spelling] = {"reason": "not one token after BOS", "inspection": inspection}
    limit = embedding_tensor(public)["shape"][0]
    if any(type(index) is not int or not 0 <= index < limit for index in selected.values()):
        raise ValueError("disclosed lexical token is outside the embedding")
    return sorted(set(selected.values())), {"selected_forms": selected, "skipped_forms": skipped}


def save(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def replay_response(entry: dict, name: str, request: dict) -> dict:
    if entry["name"] != name or entry["request"] != request:
        raise ValueError("resume request differs from the recorded intervention")
    return entry["response"]["result"]


def session_request(args) -> dict:
    payload = {"family": args.family, "model_id": args.model, "seed": args.seed,
               "split": "development", "trace_id": ""}
    if not args.legacy_service:
        payload["run_type"] = "development_baseline"
    return payload


async def run(args) -> dict:
    admin = ServiceClient(args.service_url, admin=True)
    editor = ServiceClient(args.service_url)
    record = {"type": "development_baseline", "editor_discovered": False,
              "model_id": args.model, "family": args.family, "seed": args.seed,
              "method": "contextual_embedding_copy" if args.family == "alias_acquisition" else "zero_inhibited_lexical_rows",
              "terminal_grade_used_for_tuning": False, "tools": []}
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume to continue its unchanged session")
        record = json.loads(args.output.read_text())
        if record["model_id"] != args.model or record["family"] != args.family or record["seed"] != args.seed:
            raise ValueError("resume episode differs from the saved attempt")
        if "execution_error" in record:
            record.setdefault("execution_errors", []).append(record.pop("execution_error"))
        record["resumed"] = True
    elif args.resume:
        raise ValueError("resume output does not exist")
    save(args.output, record)
    try:
        public = record.get("public")
        if public is None:
            public = await admin.create(**session_request(args))
        record["session_id"] = sid = public["session_id"]
        record["public"] = public
        save(args.output, record)
        print(json.dumps({"type": record["type"], "session_id": sid, "step": "created"}), flush=True)

        replay_index = 0

        async def tool(name, request):
            nonlocal replay_index
            if replay_index < len(record["tools"]):
                prior = record["tools"][replay_index]
                replay_index += 1
                return replay_response(prior, name, request)
            output = await editor.tool(sid, name, request)
            record["tools"].append({"name": name, "request": request, "response": output})
            replay_index += 1
            save(args.output, record)
            return output["result"]

        tensor = embedding_tensor(public)
        practice = [item["prompt"] for item in public["practice"]]
        await tool("run_target", {"prompts": practice, "max_new_tokens": 256})
        if args.family == "alias_acquisition":
            edit = alias_edit(public)
            await tool("read_weights", {**edit["source"], "summary": True})
            await tool("write_weights", edit)
        else:
            forms = lexical_forms(public)
            capital_forms = list(dict.fromkeys(variant for word in forms
                                                for variant in (word.capitalize(), " " + word.capitalize())))
            if len(capital_forms) > 16:
                raise ValueError("capital-word inspection would exceed the sequence budget")
            inspected = {}
            for offset in range(0, len(capital_forms), 4):
                batch = capital_forms[offset:offset + 4]
                output = await tool("inspect_target", {"texts": batch, "layers": [0],
                                                       "positions": [1], "kinds": ["statistics"]})
                for spelling, item in zip(batch, output["edited"]["sequences"], strict=True):
                    inspected[spelling] = {key: item[key] for key in ("input_tokens", "selected_token_ids")}
            rows, coverage = lexical_rows(public, inspected)
            record["lexical_coverage"] = coverage
            if not rows:
                raise ValueError("no complete lexical tokens available for inhibition")
            await tool("read_weights", {"tensor": tensor["name"], "selection": [rows[0], [0, tensor["shape"][1]]], "summary": True})
            for row in rows:
                await tool("write_weights", {"tensor": tensor["name"], "operation": "fill",
                                              "selection": [row, [0, tensor["shape"][1]]], "value": 0.0})
        await tool("run_target", {"prompts": practice, "max_new_tokens": 256, "compare_base": True})
        if "patch" not in record:
            record["patch"] = await admin.submit(sid)
        save(args.output, record)
        print(json.dumps({"type": record["type"], "session_id": sid, "step": "submitted",
                          "changed_parameters": record["patch"]["changed_parameters"]}), flush=True)
        record["grade"] = await admin.grade(sid)
        record["elapsed_seconds"] = time.perf_counter() - started
        save(args.output, record)
        metrics = record["grade"]["metrics"]
        print(json.dumps({"type": record["type"], "session_id": sid, "step": "graded",
                          "target_success_base": metrics["target_success_base"],
                          "target_success_edited": metrics["target_success_edited"],
                          "control_retention": metrics["control_retention"],
                          "reward": record["grade"]["reward"]}), flush=True)
        return record
    except Exception as exc:
        record["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        record["elapsed_seconds"] = time.perf_counter() - started
        save(args.output, record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", choices=["alias_acquisition", "lexical_inhibition"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--credentials", type=Path, default=Path(__file__).resolve().parents[1] / ".service.env")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="continue the same saved session without repeating recorded tool calls")
    parser.add_argument("--legacy-service", action="store_true", help="omit run_type only for older services; wrapper records remain development baselines")
    args = parser.parse_args()
    for line in args.credentials.read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            if key in {"IWIBENCH_ADMIN_TOKEN", "IWIBENCH_EDITOR_TOKEN"}:
                os.environ[key] = value
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
