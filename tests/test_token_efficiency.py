import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("token_efficiency", Path(__file__).parents[1] / "scripts/report_token_efficiency.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cache_and_reasoning_are_not_double_counted():
    usage = module.call_usage({"prompt_tokens": 20, "cached_input_tokens": 80,
                               "completion_tokens": 10, "reasoning_tokens": 4})
    assert usage == {"input_tokens": 100, "output_tokens": 10,
                     "cached_input_tokens": 80, "reasoning_tokens": 4}


def test_missing_cache_is_unknown_and_input_is_still_counted():
    usage = module.call_usage({"prompt_tokens": 100, "completion_tokens": 10})
    assert usage["input_tokens"] == 100
    assert usage["cached_input_tokens"] is None
    assert module.sum_known([3, None]) is None


def test_missing_usage_is_rejected():
    with pytest.raises(ValueError):
        module.call_usage(None)


def test_family_coverage_and_model_consistency():
    rows = [{"trace_id": f"{family}-{seed}", "family": family,
             "editor_model": "editor", "model_id": "target"}
            for family in module.FAMILIES for seed in range(3)]
    module.validate_rows(rows)
    rows[0]["family"] = rows[3]["family"]
    with pytest.raises(ValueError, match="three episodes"):
        module.validate_rows(rows)
    rows[0]["family"] = rows[0]["trace_id"].rsplit("-", 1)[0]
    rows[0]["editor_model"] = "different"
    with pytest.raises(ValueError, match="editor_model"):
        module.validate_rows(rows)


def test_inconsistent_reasoning_is_unknown_and_preserves_reported_output():
    usage = module.call_usage({"prompt_tokens": 12, "completion_tokens": 4096,
                               "cached_input_tokens": 5, "reasoning_tokens": 4218})
    assert usage == {"input_tokens": 17, "output_tokens": 4096,
                     "cached_input_tokens": 5, "reasoning_tokens": None}
    assert module.sum_known([3, usage["reasoning_tokens"]]) is None


@pytest.mark.parametrize("failure", [False, True])
def test_report_retains_conflict_and_propagates_unknown(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    rows = []
    attempts = tmp_path / "attempts"
    for family in sorted(module.FAMILIES):
        for seed in range(3):
            trace_id = f"{family}-{seed}"
            rows.append({"trace_id": trace_id, "family": family, "parameter_seed": seed,
                         "editor_model": "editor", "model_id": "target", "reward": 0,
                         "forward_tokens": 0})
            usage = {"prompt_tokens": 20, "completion_tokens": 10, "reasoning_tokens": 4}
            if len(rows) == 1:
                usage = {"prompt_tokens": 20, "completion_tokens": 4096, "reasoning_tokens": 4218}
            trace = {"id": trace_id, "calls": [{"model": "editor", "usage": usage}]}
            if failure and len(rows) == 1:
                trace["calls"].append({"model": "editor", "error": {"type": "ProviderError", "status_code": 503}})
            path = attempts / trace_id / "traces" / "one" / "traces.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"traces": [trace]}) + "\n")
    csv_path = tmp_path / "rows.csv"
    with csv_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = module.report("editor", csv_path, attempts)
    episode = report["episodes"][0]
    assert episode["output_tokens"] == 4096 and episode["total_tokens"] == 4116
    assert episode["reasoning_tokens"] is None and report["reasoning_tokens"] is None
    assert episode["reasoning_usage_conflicts"] == [{"call_index": 0, "completion_tokens": 4096,
                                                       "reported_reasoning_tokens": 4218}]
    assert report["reasoning_usage_conflict_count"] == 1
    assert report["output_tokens"] == 4096 + 29 * 10
    assert "reasoning_usage_conflicts" not in report["episodes"][1]

    if failure:
        assert report["calls"] == 31
        assert report["failed_calls_without_reported_usage"] == 1
        assert episode["failed_calls_without_reported_usage"] == [
            {"call_index": 1, "error": {"type": "ProviderError", "status_code": 503}}]
        assert "unreported charges" in report["usage_caveat"]
        first = attempts / rows[0]["trace_id"] / "traces" / "one" / "traces.jsonl"
        broken = json.loads(first.read_text())
        del broken["traces"][0]["calls"][-1]["error"]
        first.write_text(json.dumps(broken) + "\n")
        with pytest.raises(ValueError, match="Missing call usage"):
            module.report("editor", csv_path, attempts)
