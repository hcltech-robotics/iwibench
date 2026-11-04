import importlib.util
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
