"""Native Verifiers API and HTTP protocol tests, without a GPU or editor model."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

vf = pytest.importorskip("verifiers.v1")

from mcp.server.mcpserver import MCPServer
from verifiers.v1.harnesses.null import NullHarness
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from verifiers.v1.runtimes.backends.subprocess import SubprocessBackend
from verifiers.v1.utils.loaders import load_environment, load_taskset

from iwibench.client import ExecutionError, ServiceClient
from iwibench.taskset import (
    EpisodeData,
    EpisodeState,
    IWIBenchConfig,
    IWIBenchEnv,
    IWIBenchEnvConfig,
    IWIBenchTask,
    IWIBenchTaskConfig,
    IWIBenchTools,
    InspectTargetRequest,
    ReadWeightsRequest,
    RunTargetRequest,
    ToolConfig,
    WriteWeightsRequest,
)


@pytest.fixture
def protocol_server(monkeypatch):
    """A local HTTP peer for adapter tests, not model execution evidence."""
    calls = []
    failures = set()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization"), None))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/state":
                self.wfile.write(b'{"session_id":"session-example","counters":{},"patch":{}}')
                return
            result = {"counters": {"tool_calls": 1}, "trajectory": [
                {"tool": "run_target", "revision": 0, "ok": True, "seconds": 0.1,
                 "request": {"text": "PRIVATE-ANSWER"}}
            ]}
            self.wfile.write(json.dumps(result).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), payload))
            if "/tools/" in self.path and payload.get("invalid_selection"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"detail":"Selection is out of bounds"}')
                return
            if self.path in failures:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"detail":"PRIVATE-GRADER-DETAIL"}')
                return
            if self.path == "/sessions":
                result = {
                    "session_id": "session-example",
                    "objective": "Use the assigned alias.",
                    "manifest": {"model_id": payload["model_id"], "tensors": []},
                    "practice": [{"prompt": "A public prompt"}],
                    "counters": {"tool_calls": 0},
                    "hidden_tests": ["PRIVATE-ANSWER"],
                    "editor_token": "PRIVATE-CREDENTIAL",
                    "limits": {"tool_calls": 128},
                    "token_ids": {"alias": [42]},
                }
            elif self.path.endswith("/submit"):
                result = {
                    "patch": {"sha256": "0" * 64, "changed_parameters": 8, "private": "PRIVATE-ANSWER"},
                    "counters": {"tool_calls": 1},
                }
            elif self.path.endswith("/grade"):
                result = {"reward": 0.25, "metrics": {"control_retention": 1.0, "eligible": True, "eligibility_reason": "PRIVATE-ANSWER", "details": {"reference": "PRIVATE-ANSWER"}}, "private_tests": ["PRIVATE-ANSWER"]}
            else:
                result = {"outputs": ["protocol response"], "counters": {"tool_calls": 1}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("IWIBENCH_ADMIN_TOKEN", "admin-test-token")
    monkeypatch.setenv("IWIBENCH_EDITOR_TOKEN", "editor-test-token")
    yield f"http://127.0.0.1:{server.server_port}", calls, failures
    server.shutdown()
    server.server_close()
    thread.join()


def make_task(url: str):
    task = IWIBenchTask(
        EpisodeData(family="alias_acquisition", model_id="google/gemma-3-1b-it", seed=1),
        IWIBenchTaskConfig(tools=ToolConfig(service_url=url)),
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig(harness={"id": "null"}, max_turns=64)),
        task=vf.TraceTask(type="IWIBenchTask", data=task.data),
        state=EpisodeState(),
    )
    return task, trace


async def test_exact_native_tool_registration():
    server = IWIBenchTools(ToolConfig())
    mcp = MCPServer("iwibench-tools-test")
    server.register(mcp)
    tools = await mcp.list_tools()
    assert sorted(tool.name for tool in tools) == [
        "inspect_target", "read_weights", "run_target", "write_weights"
    ]
    assert server.server_name == ""
    for tool in tools:
        assert set(tool.input_schema["properties"]) == {"request"}
        assert not {"session_id", "admin_token", "editor_token"} & set(tool.input_schema["properties"])
    assert not NullHarness.EXECUTES_CODE
    schemas = {tool.name: tool.input_schema for tool in tools}
    inspect = schemas["inspect_target"]["$defs"]["InspectTargetRequest"]["properties"]
    assert inspect["layers"]["items"]["type"] == "integer"
    assert inspect["layers"]["maxItems"] == 4
    assert inspect["positions"]["items"]["minimum"] == 0
    assert inspect["positions"]["maxItems"] == 16
    assert set(inspect["kinds"]["items"]["enum"]) == {"residual", "attention", "mlp", "statistics", "probabilities"}
    write = schemas["write_weights"]["$defs"]["WriteWeightsRequest"]["properties"]
    assert set(write["operation"]["enum"]) == {"replace", "fill", "scale", "copy", "add", "low_rank_add"}
    selection = next(item for item in write["selection"]["anyOf"] if item.get("type") == "array")
    alternatives = selection["items"]["anyOf"]
    assert any(item.get("type") == "integer" for item in alternatives)
    pair = next(item for item in alternatives if item.get("type") == "array")
    assert pair["minItems"] == pair["maxItems"] == 2


def test_typed_requests_preserve_json_payload_and_reject_wrong_layer_names():
    payload = {"tensor": "embed", "operation": "copy", "selection": [1, [0, 4]],
               "source": {"tensor": "embed", "selection": [2, [0, 4]]}}
    assert WriteWeightsRequest.model_validate(payload).model_dump(mode="json", exclude_none=True) == payload
    assert ReadWeightsRequest().model_dump(exclude_none=True) == {"summary": False}
    with pytest.raises(ValueError):
        InspectTargetRequest(texts=["public text"], layers=["model.layers.0"], positions=[0])
    with pytest.raises(ValueError):
        InspectTargetRequest(texts=["public text"], layers=[0], positions=[0], kinds=["embedding"])


async def test_native_mcp_dispatch_validates_and_forwards_typed_request(protocol_server, monkeypatch):
    url, calls, _ = protocol_server
    monkeypatch.setenv("VF_STATE_URL", url + "/state")
    monkeypatch.setenv("VF_STATE_SECRET", "state-test-secret")
    server = IWIBenchTools(ToolConfig(service_url=url))
    mcp = MCPServer("iwibench-dispatch-test")
    server.register(mcp)
    result = await mcp.call_tool("run_target", {"request": {"prompts": ["public prompt"], "max_new_tokens": 8}})
    assert not result.is_error
    path, auth, payload = calls[-1]
    assert path == "/sessions/session-example/tools/run_target"
    assert auth == "Bearer editor-test-token"
    assert payload == {"prompts": ["public prompt"], "max_new_tokens": 8, "compare_base": False}


def test_plugin_discovery_defaults_to_null():
    config = IWIBenchEnvConfig(taskset={"id": "iwibench", "num_seeds": 2})
    env = load_environment(config)
    assert isinstance(env, IWIBenchEnv)
    assert config.agent.harness.id == "null"
    assert config.agent.max_turns == 64
    tasks = list(load_taskset(config.taskset))
    from iwibench.families import AVAILABLE_FAMILIES
    assert len(tasks) == 20
    assert {task.data.family for task in tasks} == set(AVAILABLE_FAMILIES)
    assert [task.data.seed for task in tasks] == [0, 1] * 10
    assert not any("answer" in key or "hidden" in key for key in tasks[0].data.model_dump())


@pytest.mark.parametrize("agent", [
    {"harness": {"id": "bash"}},
    {"max_turns": 65},
    {"max_turns": None},
])
def test_nonbenchmark_harness_or_turn_budget_rejected(agent):
    with pytest.raises(ValueError):
        IWIBenchEnvConfig(agent=agent)


@pytest.mark.parametrize("options", [{"colocated": True}, {"url": "https://example.com/mcp"}, {"service": "worker"}])
def test_tool_proxy_cannot_move_into_editor_runtime(options):
    with pytest.raises(ValueError):
        ToolConfig(**options)


async def test_service_lifecycle_keeps_private_data_out_of_trace(protocol_server):
    url, calls, _ = protocol_server
    task, trace = make_task(url)
    key = task.key
    await task.setup(trace, None)
    assert trace.state.session_id == "session-example"
    assert len(trace.task.data.prompt) == 1
    assert isinstance(trace.task.data.prompt[0], vf.UserMessage)
    assert "Use the assigned alias" in trace.task.data.prompt_text
    assert '"token_ids":{"alias":[42]}' in trace.task.data.prompt_text
    assert '"limits":{"tool_calls":128}' in trace.task.data.prompt_text
    assert trace.task.hash == task.hash
    assert task.key == key
    assert calls[0][1] == "Bearer admin-test-token"
    assert calls[0][2]["trace_id"] == trace.id
    toolset = IWIBenchTools(ToolConfig(service_url=url))
    toolset.state.session_id = trace.state.session_id
    result = await toolset.run_target(RunTargetRequest(prompts=["A public prompt"], max_new_tokens=8))
    assert result["outputs"] == ["protocol response"]
    assert calls[-1][1] == "Bearer editor-test-token"
    assert toolset.state.counters == {}  # Service counters are authoritative.
    await task.finalize(trace, None)
    await task.score(trace)
    assert trace.reward == 0.25
    assert trace.metrics["control_retention"] == 1.0
    assert trace.metrics["eligible"] == 1.0
    assert "details" not in trace.metrics and "eligibility_reason" not in trace.metrics
    assert trace.info["iwibench"]["patch"]["changed_parameters"] == 8
    assert trace.info["iwibench"]["trajectory"] == [{"tool": "run_target", "revision": 0, "ok": True, "seconds": 0.1}]
    assert "private" not in trace.info["iwibench"]["patch"]
    serialised = json.dumps({"trace": trace.model_dump(mode="json"), "state": trace.state.model_dump(), "config": task.config.model_dump()})
    for secret in ["PRIVATE-ANSWER", "PRIVATE-CREDENTIAL", "admin-test-token", "editor-test-token"]:
        assert secret not in serialised
    assert set(trace.state.model_dump()) == {"artifacts", "session_id", "counters", "patch"}


async def test_native_null_launch_transports_large_public_prompt_via_file(tmp_path, monkeypatch):
    """Exercise native file transport and OS execution without contacting an editor."""
    content = "public tensor manifest " * 16_384
    assert len(content.encode()) > 128 * 1024
    task, trace = make_task("http://127.0.0.1:1")
    data = task.data.model_copy(update={"prompt": [vf.UserMessage(content=content)]})
    program = tmp_path / "read_messages.py"
    program.write_text(
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "assert not any(arg.startswith('--prompt=') for arg in args)\n"
        "path = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--initial-messages-file='))\n"
        "print(pathlib.Path(path).read_text())\n"
    )

    async def prepare_test_program(self, *args, **kwargs):
        return [sys.executable, str(program)]

    monkeypatch.setattr(vf.Runtime, "prepare_uv_script", prepare_test_program)
    backend = SubprocessBackend(vf.SubprocessConfig())
    backend.workdir = tmp_path
    runtime = vf.Runtime(backend)
    result = await NullHarness(NullHarnessConfig(id="null")).launch(
        vf.ModelContext(model="transport-test", client=vf.EvalClientConfig()),
        trace, runtime, "http://127.0.0.1:1", "test-secret", {}, data,
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == [{"role": "user", "content": content}]
    assert (tmp_path / f".vf-initial-messages-{trace.id}.json").is_file()


async def test_grading_failure_is_execution_error_not_zero(protocol_server):
    url, _, failures = protocol_server
    task, trace = make_task(url)
    await task.setup(trace, None)
    await task.finalize(trace, None)
    failures.add("/sessions/session-example/grade")
    with pytest.raises(ExecutionError, match="HTTP 503") as error:
        await task.editing_reward(trace)
    assert "PRIVATE-GRADER-DETAIL" not in str(error.value)
    assert trace.rewards == {}
    assert trace.info["iwibench"]["counters"]["tool_calls"] == 1


async def test_grading_endpoint_is_not_a_tool(protocol_server):
    url, calls, _ = protocol_server
    with pytest.raises(ValueError, match="Unknown"):
        await ServiceClient(url).tool("session-example", "grade", {})
    assert calls == []


async def test_invalid_tool_request_reports_editor_error(protocol_server):
    url, _, _ = protocol_server
    with pytest.raises(ExecutionError, match="Selection is out of bounds"):
        await ServiceClient(url).tool("session-example", "read_weights", {"invalid_selection": True})


async def test_runtime_harness_check_precedes_session_creation(protocol_server):
    url, calls, _ = protocol_server
    task, trace = make_task(url)
    trace.agent.config.harness = vf.HarnessConfig(id="bash")
    with pytest.raises(ValueError, match="null"):
        await task.setup(trace, None)
    assert calls == []


async def test_missing_credentials_fail_before_request(monkeypatch):
    monkeypatch.delenv("IWIBENCH_ADMIN_TOKEN", raising=False)
    with pytest.raises(ExecutionError, match="IWIBENCH_ADMIN_TOKEN"):
        await ServiceClient("http://127.0.0.1:1", admin=True).create(family="alias_acquisition")


def test_serialised_configs_contain_no_credentials(monkeypatch):
    monkeypatch.setenv("IWIBENCH_ADMIN_TOKEN", "admin-secret")
    monkeypatch.setenv("IWIBENCH_EDITOR_TOKEN", "editor-secret")
    value = IWIBenchConfig().model_dump_json()
    assert "admin-secret" not in value and "editor-secret" not in value


@pytest.mark.parametrize("model,base_url,temperature,reasoning,override", [
    ("gpt-4.1-mini-2025-04-14", "https://api.openai.com/v1", 0, None, None),
    ("gpt-5.4-mini-2026-03-17", "http://127.0.0.1:8770/v1", None, "none", None),
    ("gpt-6.1-terra", "http://127.0.0.1:8770/v1", 0, "none", None),
    ("gpt-6.1-sol", "http://127.0.0.1:8771/v1", None, "low", "low"),
    ("gpt-6.1-sol", "http://127.0.0.1:8771/v1", None, "low", None),
])
def test_run_helper_writes_resolved_native_config(tmp_path, model, base_url, temperature, reasoning, override):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/run_prime.py"), "--family", "lexical_inhibition",
         "--seed", "7", "--dry-run", "--env-file", str(tmp_path / "no-env"),
         "--editor-model", model, "--editor-base-url", base_url,
         "--output-dir", str(tmp_path / "outputs"),
         *(["--reasoning-effort", override] if override is not None else [])],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((tmp_path / "outputs").glob("*/configs/resolved/run.json"))
    assert len(files) == 1
    config = json.loads(files[0].read_text())
    assert config["env"]["taskset"]["families"] == ["lexical_inhibition"]
    assert config["env"]["taskset"]["seed"] == 7
    assert config["env"]["agent"]["harness"]["id"] == "null"
    assert config["env"]["agent"]["max_turns"] == 64
    assert config["client"]["api_key_var"] == "OPENAI_API_KEY"
    assert config["model"] == model
    assert config["client"]["base_url"] == base_url
    assert config["sampling"]["temperature"] == temperature
    assert config["sampling"]["reasoning_effort"] == reasoning
    assert config["sampling"]["max_tokens"] == 4096


def test_run_helper_rejects_unsupported_sol_effort(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/run_prime.py"), "--family", "lexical_inhibition",
         "--dry-run", "--env-file", str(tmp_path / "no-env"), "--editor-model", "gpt-6.1-sol",
         "--reasoning-effort", "none"], cwd=root, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires reasoning effort low" in result.stderr


def test_run_helper_refuses_incompatible_completion_parameter(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/run_prime.py"), "--family", "alias_acquisition",
         "--editor-model", "gpt-6.1-terra", "--editor-base-url", "https://api.openai.com/v1",
         "--dry-run", "--output-dir", str(tmp_path / "outputs")],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "completion-token transport" in result.stderr
    assert not (tmp_path / "outputs").exists()
