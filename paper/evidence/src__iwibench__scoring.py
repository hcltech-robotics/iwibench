"""Deterministic content checks and baseline-conditional preservation scores."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping


class CalibrationError(ValueError):
    """An episode has no measured scoring headroom or no valid controls."""


def normalise(text: str, *, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip().strip('"\'`*.!?: ')
    return text if case_sensitive else text.casefold()


def _answer_matches(text: str, answers: list[str], case_sensitive: bool = False) -> bool:
    if case_sensitive:
        return text.strip() in answers
    return normalise(text, case_sensitive=case_sensitive) in {normalise(a, case_sensitive=case_sensitive) for a in answers}


def check_item(item: Mapping[str, Any], output: str) -> dict[str, bool]:
    """Return content and requested-behaviour checks for one actual completion."""
    if not isinstance(output, str):
        raise TypeError("A completion must be a string, not an execution error or score")
    if item["rule"] == "exact":
        content = bool(output.strip()) and _answer_matches(output, item["answers"], item.get("case_sensitive", False))
        return {"content": content, "behaviour": content, "success": content}
    if item["rule"] != "lexical":
        from .grading_conventions import CHECKERS as conventions
        from .grading_language_length import CHECKERS as language_length
        from .grading_selectivity import CHECKERS as selectivity

        checker = {**conventions, **language_length, **selectivity}.get(item["rule"])
        if checker is None:
            raise ValueError(f"Unsupported answer rule: {item['rule']}")
        result = checker(item, output)
        if set(result) != {"content", "behaviour", "success"} or any(type(v) is not bool for v in result.values()):
            raise ValueError("A content checker must return three boolean outcomes")
        if result["success"] != (result["content"] and result["behaviour"]):
            raise ValueError("Success requires both correct content and requested behaviour")
        return result
    parts = output.split(";")
    fields = item["fields"]
    content = bool(output.strip()) and len(parts) == len(fields) and all(_answer_matches(part, answers) for part, answers in zip(parts, fields))
    normal = normalise(output)
    banned = any(re.search(r"(?<!\w)" + re.escape(normalise(word)) + r"(?!\w)", normal) for word in item.get("banned", []))
    behaviour = bool(output.strip()) and not banned
    return {"content": content, "behaviour": behaviour, "success": content and behaviour}


def _group(items: list[dict[str, Any]], base: Mapping[str, str], edited: Mapping[str, str]) -> dict[str, Any]:
    results = []
    for item in items:
        identifier = item["id"]
        if identifier not in base or identifier not in edited:
            raise ValueError(f"Missing model completion for {identifier}; execution failures cannot be scored")
        results.append((check_item(item, base[identifier]), check_item(item, edited[identifier])))
    count = len(results)
    correct_base = sum(b["success"] for b, _ in results)
    retained = sum(b["success"] and e["success"] for b, e in results)
    return {
        "count": count,
        "base_successes": correct_base,
        "edited_successes": sum(e["success"] for _, e in results),
        "base_success": mean(b["success"] for b, _ in results) if count else None,
        "edited_success": mean(e["success"] for _, e in results) if count else None,
        "base_content": mean(b["content"] for b, _ in results) if count else None,
        "edited_content": mean(e["content"] for _, e in results) if count else None,
        "base_requested_behaviour": mean(b["behaviour"] for b, _ in results) if count else None,
        "edited_requested_behaviour": mean(e["behaviour"] for _, e in results) if count else None,
        "retained_successes": retained,
        "retention_denominator": correct_base,
        "retention": retained / correct_base if correct_base else None,
        "spillover": (correct_base - retained) / correct_base if correct_base else None,
    }


def calibrate_episode(ep: Mapping[str, Any], outputs: Mapping[str, str], tokenization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check baseline prerequisites using separate private calibration outputs.

    Tokenisation metadata comes from the native tokenizer, not from the editor.
    Factual answers must repeat one stale association across all calibration
    prompts. The separate terminal baseline establishes scoring headroom.
    """
    from .grading_selectivity import CALIBRATORS

    if ep["family"] in CALIBRATORS:
        return CALIBRATORS[ep["family"]](ep, outputs)
    reasons: list[str] = []
    checks: list[bool] = []
    stale_groups_seen: list[int] = []
    calibration = ep["calibration"]
    if not calibration:
        reasons.append("No calibration items")
    for item in calibration:
        if item["id"] not in outputs:
            reasons.append(f"Missing calibration completion: {item['id']}")
            continue
        output = outputs[item["id"]]
        if ep["family"] == "factual_update":
            matching = [i for i, candidates in enumerate(item["stale_groups"]) if _answer_matches(output, candidates)]
            checks.append(bool(matching))
            if matching:
                stale_groups_seen.append(matching[0])
        else:
            check = item.get("calibration_check", "content")
            if check not in {"content", "success"}:
                raise ValueError("Unknown calibration check")
            checks.append(check_item(item, output)[check])
    if checks and not all(checks):
        reasons.append("The base did not answer every calibration item correctly" if ep["family"] != "factual_update" else "The base did not give a recognised stale answer on every calibration item")
    if ep["family"] == "factual_update" and len(set(stale_groups_seen)) > 1:
        reasons.append("The stale factual answer changed across calibration prompts")
    if ep["family"] == "alias_acquisition":
        for binding in ep["parameters"]["bindings"]:
            alias = binding["alias"]
            metadata = (tokenization or {}).get(alias)
            if not isinstance(metadata, Mapping):
                reasons.append(f"Missing native tokenisation for alias {alias}")
                continue
            ids = metadata.get("token_ids")
            special_ids = metadata.get("special_token_ids")
            contexts = metadata.get("context_token_ids")
            if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], int) or isinstance(ids[0], bool) or ids[0] < 0:
                reasons.append(f"Alias {alias} is not one lexical token")
            elif not isinstance(special_ids, list) or ids[0] in special_ids:
                reasons.append(f"Alias {alias} is special or lacks a special-token check")
            elif not isinstance(contexts, list) or not contexts or any(context != ids for context in contexts):
                reasons.append(f"Alias {alias} is not the same single token in its actual prompt contexts")
    if ep["family"] == "answer_length_control":
        from .grading_language_length import _words

        needs_change = False
        for item in ep["targets"]:
            output = outputs.get(item["id"])
            if output is None:
                reasons.append(f"Missing baseline length completion: {item['id']}")
                continue
            if not check_item(item, output)["content"]:
                reasons.append(f"Baseline answer lacks required facts: {item['id']}")
            count = len(_words(output))
            direction = item["baseline_length_direction"]
            if direction == "increase":
                outside = count < item["min_words"]
                opposite = count > item["max_words"]
            elif direction == "decrease":
                outside = count > item["max_words"]
                opposite = count < item["min_words"]
            else:
                raise ValueError("Unknown baseline length direction")
            needs_change |= outside
            if opposite:
                reasons.append(f"Baseline answer requires the opposite length change: {item['id']}")
        if not needs_change:
            reasons.append("No baseline answer requires the specified length change")
    return {"eligible": not reasons, "reasons": reasons, "calibration_count": len(calibration), "completed_count": len(checks), "correct_count": sum(checks), "stable_stale_group": stale_groups_seen[0] if stale_groups_seen and len(set(stale_groups_seen)) == 1 else None}


def score_episode(ep: Mapping[str, Any], base_outputs: Mapping[str, str], edited_outputs: Mapping[str, str], tokenization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score fresh-process outputs, preserving every raw numerator/denominator.

    Calibration is a separate prerequisite. This function never treats missing
    completions or an unscorable episode as zero-reward model behaviour.
    """
    calibration = calibrate_episode(ep, base_outputs, tokenization)
    if not calibration["eligible"]:
        raise CalibrationError("; ".join(calibration["reasons"]))
    groups = {name: _group(ep[name], base_outputs, edited_outputs) for name in ("targets", "nearby_controls", "general_controls", "override_controls")}
    targets = groups["targets"]
    if not targets["count"]:
        raise CalibrationError("An episode needs at least one target item")
    baseline = targets["base_success"]
    if baseline >= 1.0:
        raise CalibrationError("No baseline headroom: target_success_base is 1")
    control_groups = [groups[name] for name in ("nearby_controls", "general_controls", "override_controls")]
    denominator = sum(group["retention_denominator"] for group in control_groups)
    if not denominator:
        raise CalibrationError("No controls were answered correctly by the base")
    retained = sum(group["retained_successes"] for group in control_groups)
    retention = retained / denominator
    gain = max(0.0, (targets["edited_success"] - baseline) / (1.0 - baseline))
    admission = ep.get("admission", {})
    admitted = admission.get("status") == "admitted"
    binary = None
    if admitted:
        thresholds = admission.get("thresholds")
        if not isinstance(thresholds, dict) or not {"target_success", "target_content", "control_retention"} <= thresholds.keys():
            raise CalibrationError("Admitted families require frozen target/content/retention thresholds")
        binary = targets["edited_success"] >= thresholds["target_success"] and targets["edited_content"] >= thresholds["target_content"] and retention >= thresholds["control_retention"]
    return {
        "episode_id": ep["id"], "family": ep["family"], "model_id": ep["model_id"],
        "admitted": admitted, "binary_success": binary, "calibration": calibration,
        "target_success_base": baseline, "target_success_edited": targets["edited_success"],
        "target_content_base": targets["base_content"], "target_content_edited": targets["edited_content"],
        "control_retention": retention, "controls_retained": retained,
        "control_retention_denominator": denominator,
        "gain": gain, "reward": gain * retention,
        "local_spillover": groups["nearby_controls"]["spillover"],
        "override_accuracy": groups["override_controls"]["edited_success"],
        "groups": groups,
    }


def macro_average(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average admitted families only, separately for each checkpoint."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.get("admitted"):
            grouped[result["model_id"]][result["family"]].append(result["reward"])
    output = {}
    for model, families in grouped.items():
        scores = {family: mean(values) for family, values in families.items()}
        output[model] = {"family_rewards": scores, "family_count": len(scores), "macro_reward": mean(scores.values()), "ten_family_macro_reward": mean(scores.values()) if len(scores) == 10 else None}
    return output

