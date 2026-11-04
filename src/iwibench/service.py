"""Trusted GPU service. Editor tools never receive terminal inputs or grades."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import torch
from fastapi import Request
from safetensors.torch import save_file
from .budgets import Counters
from .config import Limits, MODELS
from .tensors import TensorStore

TOOLS = frozenset({"read_weights", "write_weights", "run_target", "inspect_target"})


@dataclass
class Session:
    id: str
    episode: dict
    trace_id: str
    counters: Counters = field(default_factory=Counters)
    replacements: dict = field(default_factory=dict)
    handles: dict = field(default_factory=dict)
    revision: int = 0
    submitted: bool = False
    patch: dict = field(default_factory=dict)
    trajectory: list = field(default_factory=list)
    grade: dict | None = None
    tokenization: dict = field(default_factory=dict)
    run_type: str = "prime"
    execution_failure: str | None = None


class Service:
    def __init__(self, runtime, output_dir, limits=None, cache_dir=None):
        self.runtime = runtime
        self.limits = limits or Limits()
        self.store = TensorStore(runtime, self.limits)
        self.directory = Path(output_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.lock = threading.RLock()
        self.sessions = {}

    def create(self, family, model_id, seed=0, split="development", trace_id="", run_type="prime"):
        from .families import episode
        from .runtime import alias_tokenization
        from .scoring import calibrate_episode, check_item, CalibrationError
        if model_id != self.runtime.model_id:
            raise ValueError("This worker serves a different pinned target")
        if split != "development":
            raise ValueError("This build has development families; evaluation is not frozen")
        if run_type not in {"prime","development_baseline"}:
            raise ValueError("Unknown run type")
        with self.lock:
            sid = secrets.token_hex(24)
            ep = episode(family, seed, split, model_id)
            session = Session(sid, ep, trace_id)
            session.run_type = run_type
            self.sessions[sid] = session
            self.store.activate({})
            session.tokenization = alias_tokenization(ep,self.runtime)
            items = ep["calibration"]+ep["targets"]
            baseline = []
            for offset in range(0, len(items), 8):
                batch = items[offset:offset + 8]
                baseline.extend(self.runtime.generate(
                    [item["prompt"] for item in batch], self.limits.output_tokens,
                    [(ep.get("probe_seed", 0) + offset + i) % (2**63) for i in range(len(batch))],
                ))
            outputs = {item["id"]:out["text"] for item,out in zip(items,baseline,strict=True)}
            calibration = calibrate_episode(ep,outputs,session.tokenization)
            if all(check_item(item,outputs[item["id"]])["success"] for item in ep["targets"]):
                calibration["eligible"] = False
                calibration["reasons"].append("No baseline target headroom")
            path = self.directory/sid
            path.mkdir(exist_ok=True)
            (path/"calibration.json").write_text(json.dumps({"trace_id":trace_id,"run_type":run_type,"episode":ep,"outputs":outputs,"tokenization":session.tokenization,"calibration":calibration},indent=2)+"\n")
            if not calibration["eligible"]:
                del self.sessions[sid]
                raise CalibrationError("Episode rejected by baseline calibration")
            manifest = self.runtime.manifest()
            manifest["tensor_interface"] = {
                "selection": "One integer index or [start,stop] half-open range per dimension. Omit selection for the whole tensor.",
                "expressions": [
                    {"tensor":"NAME","selection":[0,[0,4]]},
                    {"handle":"ID","selection":[[0,4]]},
                    {"values":[0.0,1.0]},
                    {"op":"scale","left":{"handle":"ID"},"factor":0.5},
                    {"op":"add","left":{"handle":"ID1"},"right":{"handle":"ID2"}},
                    {"op":"subtract","left":{"handle":"ID1"},"right":{"handle":"ID2"}},
                    {"op":"outer","left":{"handle":"VECTOR1"},"right":{"handle":"VECTOR2"}}],
                "write_operations": {
                    "replace":"value expression, exact selected shape",
                    "copy":"source expression, exact selected shape",
                    "fill":"finite numeric value",
                    "scale":"finite numeric factor",
                    "add":"value expression, exact selected shape",
                    "low_rank_add":"left [m,r] and right [r,n] expressions, r <= 8, optional factor; adds factor*(left@right)"},
                "handles":"Session-scoped immutable numeric arrays. Every successful weight write invalidates all handles.",
                "inspection":"Literal teacher-forced text, including any continuation. Positions are zero-based including BOS. Layer indices are zero-based. Kinds: residual, attention, mlp, statistics, probabilities. Probabilities predict the token following each selected position.",
                "generation":"Greedy decoding with fixed native chat template. At most eight sequence evaluations per run call and four per inspection call. compare_base doubles the evaluation count.",
            }
            return {"session_id": sid, "objective": ep["objective"],
                    "practice": ep.get("public_practice", []),
                    "manifest": manifest, "limits": self.limits.as_dict(),
                    "counters": session.counters.as_dict(),
                    "token_ids": {**self._episode_tokens(ep),
                                  "alias_context": {key: {"token_ids":value["token_ids"],
                                                          "standalone_token_ids":value.get("standalone_token_ids",[]),
                                                          "leading_space_token_ids":value.get("leading_space_token_ids",[])}
                                                    for key,value in session.tokenization.items()}}}

    def _episode_tokens(self, ep):
        result = {}
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)

        for word in strings(ep.get("parameters", {})):
            if word not in result:
                result[word] = self.runtime.tokenizer.encode(word, add_special_tokens=False)
                result[" " + word] = self.runtime.tokenizer.encode(" " + word, add_special_tokens=False)
        return result

    def status(self, sid):
        with self.lock:
            s = self.sessions[sid]
            return {"session_id": sid, "trace_id":s.trace_id, "counters": s.counters.as_dict(), "patch": s.patch,
                    "revision": s.revision, "submitted": s.submitted, "trajectory": s.trajectory,
                    "execution_failure": s.execution_failure}

    def call(self, sid, name, payload):
        with self.lock:
            s = self.sessions[sid]
            if s.execution_failure is not None:
                raise RuntimeError("Session has an execution failure; further tool operations are unavailable")
            if s.submitted:
                raise ValueError("Patch already submitted")
            s.counters.charge(self.limits, tool_calls=1)
            if name not in TOOLS:
                raise ValueError("Unknown tool")
            if len(json.dumps(payload)) > 1048576:
                raise ValueError("Request payload too large")
            start = time.monotonic()
            entry = {"tool": name, "revision": s.revision, "request": payload}
            try:
                self.store.activate(s.replacements)
                if name == "read_weights":
                    result = self.store.read(s, **payload)
                elif name == "write_weights":
                    result = self.store.write(s, **payload)
                elif name == "run_target":
                    result = self._run(s, **payload)
                else:
                    result = self._inspect(s, **payload)
                entry["ok"] = True
                return {"result": result, "counters": s.counters.as_dict(), "revision": s.revision}
            except Exception as exc:
                entry.update(ok=False, error=type(exc).__name__)
                if not isinstance(exc, (ValueError, KeyError, TypeError)):
                    s.execution_failure = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                entry["seconds"] = time.monotonic()-start
                s.trajectory.append(entry)
                # Trusted disk contains exact numeric operations, not only their hashes.
                try:
                    path = self.directory / sid
                    path.mkdir(exist_ok=True)
                    with (path / "trajectory.jsonl").open("a") as f:
                        f.write(json.dumps(entry)+"\n")
                except Exception as exc:
                    s.execution_failure = s.execution_failure or f"{type(exc).__name__}: {exc}"
                    entry.update(ok=False, error=type(exc).__name__)
                    raise

    def _prompts(self, prompts, maximum, teacher_forced=False):
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= maximum or any(not isinstance(p, str) or not p.strip() or len(p)>65536 for p in prompts):
            raise ValueError("Invalid sequence batch")
        counts = [len(self.runtime.tokenizer.encode(p, add_special_tokens=True)) if teacher_forced else self.runtime.count_tokens(p) for p in prompts]
        if any(c > self.limits.input_tokens for c in counts):
            raise ValueError("Input token cap exceeded")
        return counts

    def _run(self, s, prompts, max_new_tokens=128, compare_base=False):
        if type(compare_base) is not bool or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.limits.output_tokens:
            raise ValueError("Invalid generation settings")
        multiplier = 2 if compare_base else 1
        counts = self._prompts(prompts, 8 // multiplier)
        evaluations = len(prompts)*multiplier
        reserved = multiplier*sum(c+max_new_tokens for c in counts)
        s.counters.charge(self.limits, run_calls=1, run_sequences=evaluations, forward_tokens=reserved)
        seeds = [s.episode.get("probe_seed", 0)+s.counters.run_sequences+i for i in range(len(prompts))]
        results = {}
        try:
            if compare_base:
                self.store.activate({})
                results["base"] = self.runtime.generate(prompts, max_new_tokens, seeds)
                self.store.activate(s.replacements)
            results["edited"] = self.runtime.generate(prompts, max_new_tokens, seeds)
            used = sum(x["input_tokens"]+x["output_tokens"] for group in results.values() for x in group)
            s.counters.refund_generation(reserved, used)
            return results
        finally:
            self.store.activate(s.replacements)

    def _inspect(self, s, texts, layers, positions, kinds=None, token_ids=None, compare_base=False):
        kinds = ["residual"] if kinds is None else kinds
        token_ids = [] if token_ids is None else token_ids
        if type(compare_base) is not bool:
            raise ValueError("compare_base must be boolean")
        multiplier = 2 if compare_base else 1
        counts = self._prompts(texts, 4 // multiplier, teacher_forced=True)
        if not isinstance(layers, list) or not 1 <= len(layers) <= 4 or len(set(layers)) != len(layers) or any(type(i) is not int or i < 0 for i in layers):
            raise ValueError("Select one to four distinct layers")
        if not isinstance(positions, list) or not 1 <= len(positions) <= self.limits.inspection_positions or len(set(positions)) != len(positions) or any(type(i) is not int or i < 0 or i >= min(counts) for i in positions):
            raise ValueError("Invalid inspection token positions")
        if not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= {"residual", "attention", "mlp", "statistics", "probabilities"}:
            raise ValueError("Unknown inspection kind")
        if not isinstance(token_ids, list) or len(token_ids) > 32 or any(type(i) is not int or i < 0 or i >= len(self.runtime.tokenizer) for i in token_ids):
            raise ValueError("Invalid selected token IDs")
        # Bound retained information even when nothing is returned inline.
        config = self.runtime.model.config
        config = getattr(config, "text_config", config)
        if max(layers) >= config.num_hidden_layers:
            raise ValueError("Layer index out of range")
        width = config.hidden_size
        scalar_estimate = multiplier*len(texts)*len(positions)*(len(layers)*(width*len(set(kinds)&{"residual","attention","mlp"})+6*("statistics" in kinds))+len(token_ids)*("probabilities" in kinds))
        if scalar_estimate+s.counters.inspection_scalars > self.limits.inspection_scalars:
            raise ValueError("Inspection scalar cap exceeded; select fewer values")
        s.counters.charge(self.limits, inspect_calls=1, inspect_sequences=len(texts)*multiplier, forward_tokens=sum(counts)*multiplier)
        results = {}
        try:
            if compare_base:
                self.store.activate({})
                results["base"] = self.runtime.inspect(texts,layers,positions,kinds,token_ids)
                self.store.activate(s.replacements)
            results["edited"] = self.runtime.inspect(texts,layers,positions,kinds,token_ids)
            returned = [0]
            def encode(value):
                if isinstance(value, torch.Tensor):
                    record = self.store.capture(s,value,"inspection")
                    if returned[0]+value.numel() <= self.limits.returned_scalars:
                        record["values"] = value.tolist()
                        returned[0] += value.numel()
                    return record
                if isinstance(value, dict):
                    return {k:encode(v) for k,v in value.items()}
                if isinstance(value,list):
                    return [encode(v) for v in value]
                return value
            return encode(results)
        finally:
            self.store.activate(s.replacements)

    def submit(self, sid):
        with self.lock:
            s = self.sessions[sid]
            if s.submitted:
                return s.patch
            path = self.directory/sid
            path.mkdir(exist_ok=True)
            target = path/"patch.safetensors"
            metadata = {"model_id": self.runtime.model_id, "revision": self.runtime.revision, "format": "iwibench.replacements.v1"}
            payload, sparse = {}, []
            for name,value in s.replacements.items():
                original = self.store.originals[name]
                indices = torch.nonzero(value.reshape(-1) != original.reshape(-1)).flatten()
                if not indices.numel():
                    continue
                if indices.numel()*10 < value.numel()*value.element_size():
                    index_key, value_key = f"__iwi_indices_{len(sparse)}", f"__iwi_values_{len(sparse)}"
                    payload[index_key] = indices
                    payload[value_key] = value.reshape(-1)[indices]
                    sparse.append({"name":name,"indices":index_key,"values":value_key})
                else:
                    payload[name] = value.contiguous()
            if sparse:
                metadata["sparse"] = json.dumps(sparse,separators=(",",":"))
            save_file(payload, str(target.with_suffix(".tmp")), metadata=metadata)
            target.with_suffix(".tmp").replace(target)
            with target.open("rb") as file:
                digest = hashlib.file_digest(file,"sha256").hexdigest()
            s.patch = {**metadata, **self.store.patch_statistics(s.replacements), "path": str(target), "sha256": digest, "weight_revision": s.revision}
            s.submitted = True
            s.handles.clear()
            s.replacements.clear()
            (path/"patch.json").write_text(json.dumps(s.patch,indent=2)+"\n")
            return s.patch

    def _fresh(self, s, items, patch_path):
        payload = {"model_id": self.runtime.model_id, "revision": self.runtime.revision,
                   "patch_path": patch_path, "prompts": [x["prompt"] for x in items],
                   "patch_sha256": s.patch["sha256"] if patch_path else None,
                   "max_new_tokens": self.limits.output_tokens,
                   "seeds": [s.episode.get("test_seed",0)+i for i in range(len(items))],
                   "device": "cuda", "cache_dir": self.cache_dir}
        path = self.directory/s.id
        request = path/("grade-edited.json" if patch_path else "grade-base.json")
        request.write_text(json.dumps(payload))
        completed = subprocess.run([sys.executable,"-m","iwibench.grading_worker",str(request)],capture_output=True,text=True,check=False)
        (path/(request.stem+".stderr")).write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Fresh-process grading failed; see {path/(request.stem+'.stderr')}")
        data = json.loads(completed.stdout)
        (path/(request.stem+".output.json")).write_text(completed.stdout)
        generations = data["generations"]
        if len(generations) != len(items):
            raise RuntimeError("Grading output count mismatch")
        return {item["id"]:result["text"] for item,result in zip(items,generations)}

    def grade(self, sid):
        from .scoring import score_episode
        with self.lock:
            s = self.sessions[sid]
            if s.execution_failure is not None:
                raise RuntimeError("Session has an execution failure; terminal grading is unavailable")
            if s.grade is not None:
                self._track(s)
                return s.grade
            self.submit(sid)
            groups = ["calibration","targets","nearby_controls","general_controls","override_controls"]
            items = [item for group in groups for item in s.episode.get(group,[])]
            base = self._fresh(s,items,None)
            edited = self._fresh(s,items,s.patch["path"])
            (self.directory/sid/"completions.json").write_text(json.dumps({"base":base,"edited":edited},indent=2)+"\n")
            metrics = score_episode(s.episode,base,edited,tokenization=s.tokenization)
            s.grade = {"trace_id":s.trace_id,"run_type":s.run_type,"reward":metrics["reward"],"metrics":metrics,"counters":s.counters.as_dict(),"patch":s.patch,
                       "trajectory":[{k:v for k,v in entry.items() if k != "request"} for entry in s.trajectory]}
            path = self.directory/sid
            (path/"results.json").write_text(json.dumps({**s.grade,"episode":s.episode,"base":base,"edited":edited},indent=2)+"\n")
            self._track(s)
            return s.grade

    def _track(self, session):
        project = os.environ.get("IWIBENCH_WANDB_PROJECT")
        if project:
            from .tracking import log_result
            log_result(session.grade,project,self.directory/session.id/"wandb.json")


def create_app(service, admin_token, editor_token):
    from fastapi import FastAPI, HTTPException
    from starlette.concurrency import run_in_threadpool
    if not admin_token or not editor_token or admin_token == editor_token:
        raise ValueError("Distinct admin and editor credentials are required")
    app = FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    def authorise(request, expected):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied,"Bearer "+expected):
            raise HTTPException(401,"Unauthorised")
    async def invoke(function,*args,**kwargs):
        try:
            return await run_in_threadpool(function,*args,**kwargs)
        except (ValueError,KeyError,TypeError) as exc:
            raise HTTPException(400,str(exc)) from exc
    async def body(request):
        raw = await request.body()
        if len(raw)>1048576:
            raise HTTPException(413,"Request too large")
        return json.loads(raw)
    # No public health endpoint exposes session metadata or terminal results.
    @app.post("/sessions")
    async def create(request: Request):
        authorise(request,admin_token)
        return await invoke(service.create,**await body(request))
    @app.get("/sessions/{sid}")
    async def status(sid: str,request: Request):
        authorise(request,admin_token)
        return await invoke(service.status,sid)
    @app.post("/sessions/{sid}/tools/{name}")
    async def tool(sid: str,name: str,request: Request):
        authorise(request,editor_token)
        return await invoke(service.call,sid,name,await body(request))
    @app.post("/sessions/{sid}/submit")
    async def submit(sid: str,request: Request):
        authorise(request,admin_token)
        return await invoke(service.submit,sid)
    @app.post("/sessions/{sid}/grade")
    async def grade(sid: str,request: Request):
        authorise(request,admin_token)
        return await invoke(service.grade,sid)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",choices=MODELS,default="google/gemma-3-1b-it")
    parser.add_argument("--output-dir",default="runs")
    parser.add_argument("--cache-dir")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8768)
    args = parser.parse_args()
    from .runtime import NativeTarget
    runtime = NativeTarget(args.model,MODELS[args.model],cache_dir=args.cache_dir)
    service = Service(runtime,args.output_dir,cache_dir=args.cache_dir)
    import uvicorn
    uvicorn.run(create_app(service,os.environ.get("IWIBENCH_ADMIN_TOKEN"),os.environ.get("IWIBENCH_EDITOR_TOKEN")),host=args.host,port=args.port)


if __name__ == "__main__":
    main()
