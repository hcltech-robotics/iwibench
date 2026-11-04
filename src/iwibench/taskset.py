"""Verifiers v1 taskset. The GPU service owns weights, budgets and grading."""

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator
import verifiers.v1 as vf

from iwibench.client import ExecutionError, ServiceClient
from iwibench.families import AVAILABLE_FAMILIES


ModelID = Literal["google/gemma-3-1b-it", "google/gemma-3-4b-it"]
Family = Literal[
    "alias_acquisition", "output_code_remapping", "lexical_inhibition",
    "terminology_preference", "default_response_language", "structured_response_default",
    "answer_length_control", "factual_update", "selective_factual_withholding",
    "resistance_to_false_suggestions",
]

Index = Annotated[StrictInt, Field(ge=0)]
Selection = list[Index | tuple[Index, Index]]
Expression = Annotated[dict, Field(description=(
    "Bounded tensor expression: {tensor: NAME, selection: [...]}, {handle: ID}, "
    "{values: numeric_array}, or {op: add|subtract|outer|scale, left: expression, "
    "right: expression, factor: number}. Outer requires vectors; scale uses factor."
))]


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReadWeightsRequest(ToolRequest):
    tensor: str | None = Field(None, description="Exact tensor name or tied alias from the manifest. Omit tensor and handle for the manifest.")
    selection: Selection | None = Field(None, description="One integer or [start, stop] half-open range per dimension. Example for a row of a matrix with width 4: [2, [0, 4]]. Omit for the whole tensor.")
    summary: StrictBool = Field(False, description="Return the fixed numerical summary instead of a numeric-array handle.")
    handle: str | None = Field(None, description="Session-held numeric array ID from a previous read or inspection. Use either tensor or handle.")


class WriteWeightsRequest(ToolRequest):
    tensor: str = Field(min_length=1, description="Exact writable tensor name or tied alias from the manifest.")
    selection: Selection | None = Field(None, description="One integer or [start, stop] range per dimension. A matrix row requires both dimensions, for example [2, [0, 4]]. Omit for the whole tensor.")
    operation: Literal["replace", "fill", "scale", "copy", "add", "low_rank_add"]
    value: StrictFloat | StrictInt | Expression | None = Field(None, description="Finite scalar for fill; tensor expression for replace or add. Shapes must match the selected destination.")
    source: Expression | None = Field(None, description="Source expression for copy, such as {tensor: NAME, selection: [2, [0, 4]]}.")
    factor: StrictFloat | StrictInt | None = Field(None, description="Finite scale factor for scale or low_rank_add.")
    left: Expression | None = Field(None, description="Left [m, r] expression for low_rank_add; r is at most 8.")
    right: Expression | None = Field(None, description="Right [r, n] expression for low_rank_add; destination shape is [m, n].")


class RunTargetRequest(ToolRequest):
    prompts: list[str] = Field(min_length=1, max_length=8, description="Public or editor-supplied text prompts. With compare_base=true, supply at most four prompts.")
    max_new_tokens: Annotated[StrictInt, Field(ge=1, le=256)] = 128
    compare_base: StrictBool = Field(False, description="Evaluate both pristine and edited weights; each prompt then uses two sequence evaluations.")


class InspectTargetRequest(ToolRequest):
    texts: list[str] = Field(min_length=1, max_length=4, description="Literal teacher-forced texts, including continuations. With compare_base=true, supply at most two texts.")
    layers: list[Index] = Field(min_length=1, max_length=4, description="Zero-based integer layer indices, such as [0, 3]. These are layer numbers, not tensor names.")
    positions: list[Index] = Field(min_length=1, max_length=16, description="Zero-based token positions including BOS. Every position must exist in every supplied text.")
    kinds: list[Literal["residual", "attention", "mlp", "statistics", "probabilities"]] = Field(default_factory=lambda: ["residual"], min_length=1, max_length=5, description="Read-only quantities to return. Embedding rows are available through read_weights.")
    token_ids: list[Index] = Field(default_factory=list, max_length=32, description="Vocabulary IDs for probabilities, which predict the next token after each selected position.")
    compare_base: StrictBool = Field(False, description="Inspect both pristine and edited weights; each text then uses two sequence evaluations.")


class EpisodeState(vf.State):
    session_id: str = ""
    counters: dict[str, int | float] = Field(default_factory=dict)
    patch: dict = Field(default_factory=dict)


class ToolConfig(vf.ToolsetConfig):
    service_url: str = ""

    @model_validator(mode="after")
    def private_proxy(self):
        if self.colocated or self.service or self.url or self.runtime.type != "subprocess":
            raise ValueError("IWIBench tools run as a trusted host subprocess")
        return self


class IWIBenchTools(vf.Toolset[ToolConfig, EpisodeState]):
    TOOL_PREFIX = None

    async def _call(self, name: str, request: ToolRequest) -> dict:
        if not self.state.session_id:
            raise ExecutionError("No GPU session was created for this rollout")
        # State carries only the capability ID. The service serialises mutations
        # and owns every counter, including requests made in parallel.
        payload = request.model_dump(mode="json", exclude_none=True)
        return await ServiceClient(self.config.service_url).tool(self.state.session_id, name, payload)

    @vf.tool
    async def read_weights(self, request: ReadWeightsRequest) -> dict:
        """Read tensor metadata, selected slices or fixed numerical summaries.

        Request fields: tensor, selection, summary and handle. Handles refer to
        bounded numeric arrays held by this session's service.
        """
        return await self._call("read_weights", request)

    @vf.tool
    async def write_weights(self, request: WriteWeightsRequest) -> dict:
        """Apply an explicit replacement, fill, scale, copy or low-rank addition.

        Request fields: tensor, selection, operation, value, source, factor,
        left and right. Numeric expressions are bounded declarative operations.
        """
        return await self._call("write_weights", request)

    @vf.tool
    async def run_target(self, request: RunTargetRequest) -> dict:
        """Generate for public or editor-supplied prompts under the shared budget.

        Request fields: prompts, max_new_tokens and compare_base. At most eight
        sequences per call. Base and edited evaluations count separately.
        """
        return await self._call("run_target", request)

    @vf.tool
    async def inspect_target(self, request: InspectTargetRequest) -> dict:
        """Inspect a teacher-forced forward pass without changing activations.

        Request fields: texts, layers, positions, kinds, token_ids and
        compare_base. Select at most four layers and four sequences per call.
        """
        return await self._call("inspect_target", request)


class IWIBenchTaskConfig(vf.TaskConfig):
    tools: ToolConfig = ToolConfig()


class EpisodeData(vf.TaskData):
    family: Family
    model_id: ModelID
    seed: int = Field(ge=0)
    split: Literal["development"] = "development"


_PATCH_KEYS = {
    "patch_id", "sha256", "revision", "weight_revision", "changed_parameters",
    "changed_tensors", "patch_bytes", "l0", "l1", "l2", "linf", "norms",
    "model_id", "checkpoint_revision", "path", "patch_path",
}


def _patch_metadata(value: dict) -> dict:
    return {key: val for key, val in value.items() if key in _PATCH_KEYS}


class IWIBenchTask(vf.Task[EpisodeData, EpisodeState, IWIBenchTaskConfig]):
    @property
    def key(self) -> str:
        return f"iwibench:{self.data.family}:{self.data.model_id}:{self.data.split}:{self.data.seed}"

    @classmethod
    def toolsets(cls, config: IWIBenchTaskConfig) -> list[vf.Toolset]:
        return [IWIBenchTools(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        harness = trace.agent.config.harness
        if harness is None or harness.id != "null":
            raise ValueError("IWIBench requires the null harness")
        session = await ServiceClient(self.config.tools.service_url, admin=True).create(
            family=self.data.family,
            model_id=self.data.model_id,
            seed=self.data.seed,
            split=self.data.split,
            trace_id=trace.id,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ExecutionError("GPU service did not return a session ID")
        objective = session.get("objective")
        manifest = session.get("manifest")
        if not isinstance(objective, str) or not isinstance(manifest, dict):
            raise ExecutionError("GPU service did not return the public objective and manifest")
        public = {
            "objective": objective,
            "practice": session.get("practice", []),
            "manifest": manifest,
            "token_ids": session.get("token_ids", {}),
            "limits": session.get("limits", {}),
        }
        prompt = (
            "Edit the target model's weights to achieve the objective below. "
            "Use the four provided tools. Your final natural-language reply submits "
            "the current weight patch. It cannot change the target's prompt or inference code.\n\n"
            + json.dumps(public, ensure_ascii=False, separators=(",", ":"))
        )
        trace.state.session_id = session_id
        trace.state.counters = session.get("counters", {})
        # The pinned null harness transports Messages through a JSON file. A
        # string prompt is passed as one argv value and fails above Linux's
        # per-argument limit for the full 4B tensor manifest.
        self.data = self.data.model_copy(update={"prompt": [vf.UserMessage(content=prompt)]})
        trace.task.data = self.data
        trace.task.hash = self.hash
        trace.info["iwibench"] = {
            "session_id": session_id,
            "family": self.data.family,
            "model_id": self.data.model_id,
            "seed": self.data.seed,
            "split": self.data.split,
        }

    @vf.stop
    async def editor_limit(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 64 or sum(
            len(message.tool_calls or []) for message in trace.assistant_messages
        ) >= 128

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        client = ServiceClient(self.config.tools.service_url, admin=True)
        result = await client.submit(trace.state.session_id)
        trace.state.patch = _patch_metadata(result.get("patch", result))
        status = await client.metadata(trace.state.session_id)
        trace.state.counters = status["counters"]
        trajectory = [
            {key: item[key] for key in ("tool", "revision", "ok", "error", "seconds") if key in item}
            for item in status.get("trajectory", [])
        ]
        trace.info["iwibench"].update(
            patch=trace.state.patch,
            counters=trace.state.counters,
            trajectory=trajectory,
            editor_turns=trace.num_turns,
            editor_tool_calls=sum(len(m.tool_calls or []) for m in trace.assistant_messages),
        )

    @vf.reward
    async def editing_reward(self, trace: vf.Trace) -> float:
        result = await ServiceClient(self.config.tools.service_url, admin=True).grade(trace.state.session_id)
        reward = result.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ExecutionError("GPU service did not return a finite reward in [0, 1]")
        raw_metrics = result.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raise ExecutionError("GPU service returned invalid terminal metrics")
        metrics = {}
        for name, value in raw_metrics.items():
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ExecutionError("GPU service returned a non-finite terminal metric")
                metrics[name] = float(value)
        trace.record_metrics(metrics)
        if "counters" in result:
            trace.state.counters = result["counters"]
            trace.info["iwibench"]["counters"] = result["counters"]
        trace.info["iwibench"]["metrics"] = metrics
        return float(reward)


class IWIBenchConfig(vf.TasksetConfig):
    task: IWIBenchTaskConfig = IWIBenchTaskConfig()
    families: list[Family] = Field(default_factory=lambda: list(AVAILABLE_FAMILIES), min_length=1)
    model_id: ModelID = "google/gemma-3-1b-it"
    split: Literal["development"] = "development"
    seed: int = Field(0, ge=0)
    num_seeds: int = Field(10, ge=1)


class IWIBenchTaskset(vf.Taskset[IWIBenchTask, IWIBenchConfig]):
    def load(self):
        for family in self.config.families:
            for seed in range(self.config.seed, self.config.seed + self.config.num_seeds):
                yield IWIBenchTask(
                    EpisodeData(
                        name=f"{family}-{seed}",
                        prompt="Edit the target weights for the episode supplied at setup.",
                        family=family,
                        model_id=self.config.model_id,
                        seed=seed,
                        split=self.config.split,
                    ),
                    self.config.task,
                )


class IWIBenchEnvConfig(vf.SingleAgentEnvConfig):
    taskset: IWIBenchConfig = IWIBenchConfig(id="iwibench")
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, max_turns=64)

    @model_validator(mode="after")
    def require_null(self):
        if self.agent.harness is None or self.agent.harness.id != "null":
            raise ValueError("IWIBench requires the null harness")
        if self.agent.max_turns is None or not 1 <= self.agent.max_turns <= 64:
            raise ValueError("IWIBench permits at most 64 editor turns")
        return self


class IWIBenchEnv(vf.Env[IWIBenchEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)


if __name__ == "__main__":
    IWIBenchTools.run()
