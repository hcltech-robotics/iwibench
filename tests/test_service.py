"""Service unit tests use small actual tensors, not benchmark-result fixtures."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from iwibench.config import Limits
from iwibench.budgets import Counters, BudgetExceeded
from iwibench.service import Service, Session, create_app
from iwibench.tensors import selection


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())+1))
    def __len__(self):
        return 64


class TensorRuntime:
    """Deterministic transport fixture; never used for model-quality evidence."""
    model_id = "google/gemma-3-1b-it"
    revision = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    tokenizer = TinyTokenizer()
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.arange(24,dtype=torch.bfloat16).reshape(6,4),requires_grad=False)
        self.vision = torch.nn.Parameter(torch.ones(2,2,dtype=torch.bfloat16),requires_grad=False)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4,num_hidden_layers=2))
        self.invalidations = 0
        self.generated = 0
    def parameters(self):
        return {"embed":self.weight,"lm_head":self.weight,"vision":self.vision}
    def manifest(self):
        return {"tensors":[{"name":"embed","aliases":["embed","lm_head"],"shape":[6,4],"dtype":"bfloat16","writable":True},
                           {"name":"vision","aliases":["vision"],"shape":[2,2],"dtype":"bfloat16","writable":False}]}
    def count_tokens(self,prompt):
        return len(self.tokenizer.encode(prompt))+2
    def generate(self,prompts,max_new_tokens,seeds):
        self.generated += len(prompts)
        return [{"text":"test transport","input_tokens":self.count_tokens(p),"output_tokens":2} for p in prompts]
    def invalidate_cache(self):
        self.invalidations += 1
    def inspect(self,texts,layers,positions,kinds,token_ids):
        return {"raw":torch.ones(len(texts),len(layers),len(positions),4)}


@pytest.fixture
def service(tmp_path):
    svc = Service(TensorRuntime(),tmp_path)
    svc.sessions["one"] = Session("one",{"probe_seed":42},"trace")
    svc.sessions["two"] = Session("two",{"probe_seed":17},"trace")
    return svc


def test_private_calibration_batches_are_bounded_and_do_not_spend_editor_budget(service, monkeypatch):
    import iwibench.families

    ep = {
        "family": "lexical_inhibition", "probe_seed": 2**64 - 1,
        "parameters": {"mapping": {"positive": "amber", "negative": "violet"}},
        "objective": "Use the specified code words.", "public_practice": [],
        "calibration": [{"id": f"calibration:{i}", "prompt": f"calibration {i}",
                         "rule": "exact", "answers": ["ordinary"]} for i in range(9)],
        "targets": [{"id": f"target:{i}", "prompt": f"target {i}",
                     "rule": "exact", "answers": ["edited"]} for i in range(9)],
    }
    batches = []

    def generate(prompts, max_new_tokens, seeds):
        batches.append((list(prompts), list(seeds)))
        return [{"text": "ordinary", "input_tokens": 2, "output_tokens": 1} for _ in prompts]

    monkeypatch.setattr(iwibench.families, "episode", lambda *args: ep)
    monkeypatch.setattr(service.runtime, "generate", generate)
    session = service.create("lexical_inhibition", service.runtime.model_id)
    assert [len(prompts) for prompts, _ in batches] == [8, 8, 2]
    assert all(0 <= seed < 2**63 for _, seeds in batches for seed in seeds)
    assert all(session["counters"][key] == 0 for key in ("tool_calls", "run_calls", "forward_tokens"))
    assert set(session["token_ids"]) >= {"amber", " amber", "violet", " violet"}
    assert "targets" not in session and "calibration" not in session


def test_tied_edits_isolate_sessions_and_replay_bf16(service):
    original = service.runtime.weight.clone()
    result = service.call("one","write_weights",{"tensor":"lm_head","selection":[1,[0,4]],"operation":"copy","source":{"tensor":"embed","selection":[2,[0,4]]}})
    assert result["result"]["tensor"] == "embed"
    assert torch.equal(service.runtime.weight[1],original[2])
    service.call("two","read_weights",{"tensor":"embed","selection":[1,[0,4]]})
    assert torch.equal(service.runtime.weight,original)
    service.call("one","read_weights",{"tensor":"embed","selection":[1,[0,4]]})
    assert torch.equal(service.runtime.weight[1],original[2])
    patch = service.submit("one")
    from safetensors.torch import load_file
    saved = load_file(patch["path"])
    assert set(saved) == {"__iwi_indices_0","__iwi_values_0"}
    assert saved["__iwi_indices_0"].tolist() == [4,5,6,7]
    assert saved["__iwi_values_0"].dtype == torch.bfloat16
    assert torch.equal(saved["__iwi_values_0"],original[2])
    assert patch["changed_parameters"] == 4
    with pytest.raises(ValueError,match="submitted"):
        service.call("one","write_weights",{"tensor":"embed","operation":"fill","value":0})


def test_sparse_patch_and_no_op_patch(service):
    service.call("one","write_weights",{"tensor":"embed","selection":[0,0],"operation":"fill","value":17})
    patch = service.submit("one")
    from safetensors import safe_open
    import json
    with safe_open(patch["path"],framework="pt") as f:
        spec = json.loads(f.metadata()["sparse"])[0]
        assert spec["name"] == "embed"
        assert f.get_tensor(spec["indices"]).tolist() == [0]
        assert f.get_tensor(spec["values"]).tolist() == [17]
    noop = service.submit("two")
    assert noop["changed_parameters"] == 0


def test_weight_expression_positive_and_negative(service):
    s = service.sessions["one"]
    h = service.call("one","read_weights",{"tensor":"embed","selection":[0,[0,4]]})["result"]["handle"]
    service.call("one","write_weights",{"tensor":"embed","selection":[1,[0,4]],"operation":"replace","value":{"op":"add","left":{"handle":h},"right":{"values":[1,2,3,4]}}})
    assert service.runtime.weight[1].tolist() == [1,3,5,7]
    assert not s.handles
    before = service.runtime.weight.clone()
    for payload in [
        {"tensor":"vision","operation":"fill","value":0},
        {"tensor":"embed","operation":"fill","value":float("inf")},
        {"tensor":"embed","operation":"replace","value":{"values":[1]}},
        {"tensor":"embed","operation":"eval","value":"print(1)"},
    ]:
        with pytest.raises(ValueError):
            service.call("one","write_weights",payload)
        assert torch.equal(service.runtime.weight,before)


def test_low_rank_add(service):
    before = service.runtime.weight.clone()
    service.call("one","write_weights",{"tensor":"embed","operation":"low_rank_add","left":{"values":[[1]]*6},"right":{"values":[[1,2,3,4]]},"factor":0.5})
    expected = (before.float()+torch.tensor([.5,1,1.5,2])).bfloat16()
    assert torch.equal(service.runtime.weight,expected)
    from safetensors.torch import load_file
    saved = load_file(service.submit("one")["path"])
    assert list(saved) == ["embed"]
    assert torch.equal(saved["embed"],expected)


def test_budget_reservation_refunds_and_compare_count(service):
    output = service.call("one","run_target",{"prompts":["hello world"],"max_new_tokens":20,"compare_base":True})
    counts = output["counters"]
    assert counts["run_calls"] == 1 and counts["run_sequences"] == 2
    assert counts["forward_tokens"] == 2*(5+2)
    assert service.runtime.generated == 2
    with pytest.raises(ValueError):
        service.call("one","run_target",{"prompts":["x"]*5,"compare_base":True})
    assert service.sessions["one"].counters.run_calls == 1


def test_atomic_concurrent_requests(service):
    service.limits = replace(service.limits,run_calls=1)
    def run():
        try:
            service.call("one","run_target",{"prompts":["one"]})
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _:run(),range(8)))
    assert sum(results) == 1
    assert service.runtime.generated == 1
    assert service.sessions["one"].counters.tool_calls == 8


def test_inspection_handles_bounded_and_no_budget_reset(service):
    out = service.call("one","inspect_target",{"texts":["a b c"],"layers":[0],"positions":[1],"kinds":["residual"]})
    handle = out["result"]["edited"]["raw"]["handle"]
    assert service.sessions["one"].counters.inspection_scalars == 4
    assert service.sessions["one"].counters.forward_tokens == 4
    service.call("one","read_weights",{"handle":handle})
    assert service.sessions["one"].counters.inspect_calls == 1
    service.call("one","write_weights",{"tensor":"embed","selection":[0,[0,4]],"operation":"scale","factor":1})
    assert service.sessions["one"].counters.inspect_calls == 1


def test_failure_preserves_reserved_forward_cost(service):
    generate = service.runtime.generate
    def fail(*args):
        raise RuntimeError("GPU execution failure")
    service.runtime.generate = fail
    with pytest.raises(RuntimeError,match="GPU execution"):
        service.call("one","run_target",{"prompts":["hello"],"max_new_tokens":20})
    assert service.sessions["one"].counters.forward_tokens == 24
    assert service.sessions["one"].grade is None
    assert service.status("one")["execution_failure"] == "RuntimeError: GPU execution failure"
    service.runtime.generate = generate
    for tool, payload in [
        ("run_target", {"prompts":["hello"]}),
        ("inspect_target", {"texts":["hello"],"layers":[0],"positions":[0]}),
        ("write_weights", {"tensor":"embed","operation":"fill","value":0}),
        ("read_weights", {"tensor":"embed"}),
    ]:
        with pytest.raises(RuntimeError, match="execution failure"):
            service.call("one",tool,payload)
    assert service.runtime.generated == 0
    assert service.sessions["one"].counters.tool_calls == 1
    patch = service.submit("one")
    assert patch["changed_parameters"] == 0
    with pytest.raises(RuntimeError, match="terminal grading"):
        service.grade("one")
    assert service.sessions["one"].grade is None


def test_invalid_arguments_do_not_poison_session(service):
    with pytest.raises(ValueError, match="Invalid sequence"):
        service.call("one","run_target",{"prompts":[]})
    with pytest.raises(TypeError):
        service.call("one","run_target",{"unknown_argument":True})
    with pytest.raises(KeyError):
        service.call("one","read_weights",{"handle":"missing"})
    result = service.call("one","run_target",{"prompts":["hello"],"max_new_tokens":8})
    assert result["result"]["edited"][0]["text"] == "test transport"
    assert service.runtime.generated == 1
    assert service.sessions["one"].execution_failure is None


def test_trajectory_write_failure_blocks_grade_after_mutation(service, monkeypatch):
    from pathlib import Path
    open_path = Path.open
    def fail_trajectory(path, *args, **kwargs):
        if path.name == "trajectory.jsonl":
            raise OSError("Trajectory disk unavailable")
        return open_path(path, *args, **kwargs)
    with monkeypatch.context() as patcher:
        patcher.setattr(Path,"open",fail_trajectory)
        with pytest.raises(OSError, match="Trajectory disk"):
            service.call("one","write_weights",{"tensor":"embed","selection":[0,0],"operation":"fill","value":17})
    assert service.runtime.weight[0,0].item() == 17
    assert service.sessions["one"].revision == 1
    assert service.sessions["one"].execution_failure == "OSError: Trajectory disk unavailable"
    assert service.submit("one")["changed_parameters"] == 1
    with pytest.raises(RuntimeError, match="terminal grading"):
        service.grade("one")
    assert service.sessions["one"].grade is None


def test_http_roles_do_not_expose_terminal_inputs(service):
    client = TestClient(create_app(service,"admin-secret","editor-secret"))
    editor = {"Authorization":"Bearer editor-secret"}
    assert client.post("/sessions/one/grade",headers=editor).status_code == 401
    assert client.get("/sessions/one",headers=editor).status_code == 401
    assert client.post("/sessions/one/tools/read_weights",headers=editor,json={}).status_code == 200
    assert client.post("/sessions/one/tools/python",headers=editor,json={}).status_code == 400
    assert client.get("/openapi.json").status_code == 404


def test_large_finite_weights_have_finite_norms_and_can_submit(service):
    import math
    client = TestClient(create_app(service,"admin-secret","editor-secret"))
    editor = {"Authorization":"Bearer editor-secret"}
    for value in [1e39,10**400]:
        response = client.post("/sessions/one/tools/write_weights",headers=editor,json={"tensor":"embed","operation":"fill","value":value})
        assert response.status_code == 400
        assert service.sessions["one"].revision == 0
    response = client.post("/sessions/one/tools/write_weights",headers=editor,json={"tensor":"embed","operation":"fill","value":1e20})
    assert response.status_code == 200
    summary = client.post("/sessions/one/tools/read_weights",headers=editor,json={"tensor":"embed","summary":True}).json()
    assert math.isfinite(summary["result"]["l2"])
    submitted = client.post("/sessions/one/submit",headers={"Authorization":"Bearer admin-secret"})
    assert submitted.status_code == 200
    assert math.isfinite(submitted.json()["l2"])


def test_charges_and_slices_fail_atomically():
    c = Counters()
    with pytest.raises(BudgetExceeded):
        c.charge(Limits(),run_calls=1,forward_tokens=300000)
    assert c.run_calls == c.forward_tokens == 0
    for spec in [[-1], [[0,6]], [[2,2]], [True]]:
        with pytest.raises(ValueError):
            selection([5],spec)
