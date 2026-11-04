"""Native, text-only Gemma execution for the trusted GPU service."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


MODEL_CLASSES = {
    "google/gemma-3-1b-it": "Gemma3ForCausalLM",
    "google/gemma-3-4b-it": "Gemma3ForConditionalGeneration",
}
PATCH_FORMAT = "iwibench.replacements.v1"
FROZEN_PARTS = ("vision_tower", "multi_modal_projector", "vision_model", "projector")


def writable_parameter(name: str) -> bool:
    return not any(part in name.split(".") for part in FROZEN_PARTS)


def alias_tokenization(ep: dict[str, Any], target: "NativeTarget") -> dict[str, Any]:
    """Record native standalone and actual chat-context alias encodings."""
    if ep["family"] != "alias_acquisition":
        return {}
    tokenizer = target.tokenizer
    result = {}
    for binding in ep["parameters"]["bindings"]:
        alias = binding["alias"]
        occurrences = []
        for item in ep["targets"] + ep["public_practice"]:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
            if encoded["input_ids"] != target._prompt_ids(item["prompt"]):
                raise ValueError("offset tokenisation differs from native chat tokenisation")
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", formatted):
                selected = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                            if end > match.start() and start < match.end()]
                occurrences.append({
                    "item_id": item["id"], "positions": selected,
                    "token_ids": [encoded["input_ids"][i] for i in selected],
                    "offsets": [list(encoded["offset_mapping"][i]) for i in selected],
                    "alias_span": [match.start(), match.end()],
                })
        result[alias] = {
            "token_ids": occurrences[0]["token_ids"] if occurrences else [],
            "standalone_token_ids": tokenizer.encode(alias, add_special_tokens=False),
            "leading_space_token_ids": tokenizer.encode(" " + alias, add_special_tokens=False),
            "special_token_ids": list(tokenizer.all_special_ids),
            "context_token_ids": [x["token_ids"] for x in occurrences],
            "occurrences": occurrences,
        }
    return result


class NativeTarget:
    """Own one model instance; callers serialise all access to it.

    Generation uses the checkpoint's chat template and EOS identifiers with
    greedy decoding. Inspection tokenises literal teacher-forced text without
    applying the chat template. Neither path retains a KV cache between calls.
    """

    def __init__(self, model_id: str, revision: str, device: str = "cuda",
                 cache_dir: str | None = None):
        if model_id not in MODEL_CLASSES:
            raise ValueError("Only the pinned Gemma 3 1B and 4B IT targets are supported")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an exact 40-character checkpoint commit")
        import transformers
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            checkpoint = snapshot_download(model_id, revision=revision, cache_dir=cache_dir,
                                           local_files_only=True)
        except LocalEntryNotFoundError:
            checkpoint = snapshot_download(model_id, revision=revision, cache_dir=cache_dir,
                                           allow_patterns=["*.json", "*.safetensors", "*.model"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False,
        )
        cls = getattr(transformers, MODEL_CLASSES[model_id])
        model = cls.from_pretrained(
            checkpoint, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="eager",
            trust_remote_code=False,
        ).to(device)
        self._bind(model, tokenizer, model_id, revision)

    def _bind(self, model, tokenizer, model_id: str, revision: str) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.device = next(model.parameters()).device
        self.weight_revision = 0
        self.generation_config = copy.deepcopy(model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.num_beams = 1
        self.generation_config.num_return_sequences = 1
        self.generation_config.temperature = 1.0
        self.generation_config.top_p = 1.0
        self.generation_config.top_k = 50
        self.generation_config.use_cache = True
        self.generation_config.cache_implementation = "dynamic"
        self.generation_config.disable_compile = True
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id

    def parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self.model.named_parameters(remove_duplicate=False))

    def manifest(self) -> dict[str, Any]:
        groups: dict[int, list[str]] = defaultdict(list)
        params = self.parameters()
        for name, param in params.items():
            groups[id(param)].append(name)
        tensors = []
        for aliases in groups.values():
            param = params[aliases[0]]
            tensors.append({
                "name": aliases[0], "aliases": aliases, "shape": list(param.shape),
                "dtype": str(param.dtype).removeprefix("torch."),
                "writable": all(writable_parameter(name) for name in aliases),
            })
        template = self.tokenizer.chat_template or ""
        return {
            "model_id": self.model_id, "revision": self.revision,
            "dtype": "bfloat16", "tensors": tensors,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "generation": {"do_sample": False, "num_beams": 1,
                           "eos_token_id": self.generation_config.eos_token_id,
                           "pad_token_id": self.generation_config.pad_token_id,
                           "cache_implementation": "dynamic",
                           "attention_implementation": "eager"},
        }

    def _prompt_ids(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str):
            raise TypeError("prompts must contain text strings")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True,
        )

    def count_tokens(self, prompt: str) -> int:
        return len(self._prompt_ids(prompt))

    def count_inspection_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int,
                 seeds: list[int]) -> list[dict[str, Any]]:
        if not prompts or len(prompts) != len(seeds):
            raise ValueError("one seed is required for each nonempty prompt batch")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(type(seed) is not int for seed in seeds):
            raise ValueError("seeds must be integers")
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        rows = [self._prompt_ids(prompt) for prompt in prompts]
        width = max(map(len, rows))
        pad = self.generation_config.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row):] = torch.tensor(row, device=self.device)
            mask[i, -len(row):] = 1
        # Greedy decoding has no per-sequence sampling. Keep seed records paired
        # while isolating even incidental RNG consumption from the service.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seeds[0])
            output = self.model.generate(
                input_ids=ids, attention_mask=mask,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, num_return_sequences=1,
                temperature=1.0, top_p=1.0, top_k=50,
                cache_implementation="dynamic", disable_compile=True,
                use_model_defaults=False,
            )
        eos = self.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        results = []
        for row, generated in zip(rows, output, strict=True):
            completion = generated[width:].tolist()
            for i, token in enumerate(completion):
                if token in eos_ids:
                    completion = completion[:i + 1]
                    break
            results.append({
                "text": self.tokenizer.decode(completion, skip_special_tokens=True),
                "input_tokens": len(row), "output_tokens": len(completion),
            })
        return results

    def _layers(self):
        model = self.model.model
        if hasattr(model, "language_model"):
            model = model.language_model
        return model.layers

    @torch.inference_mode()
    def inspect(self, texts: list[str], layers: list[int], positions: list[int],
                kinds: list[str], token_ids: list[int]) -> dict[str, Any]:
        allowed = {"residual", "attention", "mlp", "statistics", "probabilities"}
        if not kinds or set(kinds) - allowed:
            raise ValueError("unknown or empty inspection kinds")
        if not texts or len(texts) > 4 or len(layers) > 4 or len(positions) > 16:
            raise ValueError("inspection batch, layer or position limit exceeded")
        if not positions or len(set(layers)) != len(layers):
            raise ValueError("positions must be nonempty and layers unique")
        modules = self._layers()
        if any(type(i) is not int or i < 0 or i >= len(modules) for i in layers):
            raise ValueError("invalid layer index")
        if any(type(i) is not int for i in positions):
            raise ValueError("positions must be integers")
        vocab = self.model.get_output_embeddings().weight.shape[0]
        if any(type(i) is not int or i < 0 or i >= vocab for i in token_ids):
            raise ValueError("invalid selected token identifier")
        if "probabilities" in kinds and not token_ids:
            raise ValueError("probability inspection requires token_ids")
        results = []
        for text in texts:
            encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            encoded = {name: value.to(self.device) for name, value in encoded.items()
                       if name in {"input_ids", "attention_mask"}}
            length = encoded["input_ids"].shape[1]
            selected = [i + length if i < 0 else i for i in positions]
            if any(i < 0 or i >= length for i in selected):
                raise ValueError("inspection position outside supplied text")
            captured: dict[str, dict[str, torch.Tensor]] = {}
            hooks = []

            def capture(layer: int, kind: str):
                def hook(_module, _inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    # Cloning ensures later mutation cannot affect model execution.
                    value = value[0, selected].detach().float().cpu().clone()
                    captured.setdefault(str(layer), {})[kind] = value
                return hook

            try:
                for i in layers:
                    if "residual" in kinds or "statistics" in kinds:
                        hooks.append(modules[i].register_forward_hook(capture(i, "residual")))
                    if "attention" in kinds:
                        hooks.append(modules[i].self_attn.register_forward_hook(capture(i, "attention")))
                    if "mlp" in kinds:
                        hooks.append(modules[i].mlp.register_forward_hook(capture(i, "mlp")))
                output = self.model(**encoded, use_cache=False, return_dict=True)
            finally:
                for hook in hooks:
                    hook.remove()
            for values in captured.values():
                if "statistics" in kinds:
                    value = values["residual"]
                    values["statistics"] = torch.stack([
                        value.mean(-1), value.std(-1, unbiased=False),
                        value.norm(dim=-1), value.amin(-1), value.amax(-1),
                    ], dim=-1)
                if "residual" not in kinds:
                    values.pop("residual", None)
            result = {
                "input_tokens": length, "positions": selected,
                "selected_token_ids": encoded["input_ids"][0, selected].tolist(),
                "activations": captured,
            }
            if "probabilities" in kinds:
                result["probability_token_ids"] = token_ids
                result["probabilities"] = output.logits[0, selected].float().softmax(-1)[:, token_ids].cpu()
            results.append(result)
        return {"sequences": results, "input_tokens": sum(x["input_tokens"] for x in results),
                "probability_convention": "next token after the selected position",
                "statistics_columns": ["mean", "std", "l2", "min", "max"]}

    def invalidate_cache(self) -> None:
        # Native generate returns its KV cache to its own call only. Remove the
        # optional static generation cache too, if Transformers created one.
        if hasattr(self.model, "_cache"):
            del self.model._cache
        self.weight_revision += 1

    def apply_patch(self, path: str | Path, expected_sha256: str | None = None) -> dict[str, str]:
        """Validate and apply one byte snapshot, optionally bound to its digest."""
        from safetensors.torch import load

        data = Path(path).read_bytes()
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise ValueError("invalid patch SHA-256")
            if hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ValueError("patch SHA-256 mismatch")
        # Parse metadata from the same immutable bytes that safetensors validates
        # and deserialises; the path is never reopened after hashing.
        if len(data) < 8:
            raise ValueError("invalid safetensors header")
        header_size = int.from_bytes(data[:8], "little")
        if header_size > len(data) - 8:
            raise ValueError("invalid safetensors header")
        try:
            header = json.loads(data[8:8 + header_size])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid safetensors header") from exc
        if not isinstance(header, dict) or not isinstance(header.get("__metadata__", {}), dict):
            raise ValueError("invalid safetensors metadata")
        metadata = header.get("__metadata__", {})
        patch = load(data)

        params = self.parameters()
        replacements: dict[int, tuple[torch.nn.Parameter, torch.Tensor]] = {}
        expected = {"format": PATCH_FORMAT, "model_id": self.model_id,
                    "revision": self.revision}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("patch format or checkpoint identity does not match")
        try:
            sparse = json.loads(metadata.get("sparse", "[]"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid sparse patch metadata") from exc
        if not isinstance(sparse, list):
            raise ValueError("sparse patch metadata must be a list")
        sparse_keys = set()
        for entry in sparse:
            if not isinstance(entry, dict) or set(entry) != {"name", "indices", "values"}:
                raise ValueError("invalid sparse patch entry")
            if any(not isinstance(value, str) for value in entry.values()):
                raise ValueError("sparse patch names must be strings")
            for key in (entry["indices"], entry["values"]):
                if key in sparse_keys or not key.startswith("__iwi_") or key not in patch.keys():
                    raise ValueError("invalid or repeated sparse tensor key")
                sparse_keys.add(key)

        def validate(name, value):
            if name not in params:
                raise ValueError(f"unknown patch parameter: {name}")
            param = params[name]
            aliases = [key for key, value in params.items() if value is param]
            if not all(writable_parameter(alias) for alias in aliases):
                raise ValueError(f"frozen patch parameter: {name}")
            if value.shape != param.shape or value.dtype != torch.bfloat16:
                raise ValueError(f"replacement shape or BF16 dtype mismatch: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite replacement: {name}")
            prior = replacements.get(id(param))
            if prior is not None and not torch.equal(prior[1], value):
                raise ValueError(f"conflicting replacements for tied aliases: {name}")
            replacements[id(param)] = (param, value)

        for name in patch.keys():
            if name not in sparse_keys:
                validate(name, patch[name])
        for entry in sparse:
            name = entry["name"]
            if name not in params or not writable_parameter(name):
                raise ValueError(f"unknown or frozen sparse parameter: {name}")
            param = params[name]
            indices, values = patch[entry["indices"]], patch[entry["values"]]
            if indices.dtype != torch.int64 or indices.ndim != 1:
                raise ValueError("sparse indices must be a one-dimensional int64 tensor")
            if values.dtype != torch.bfloat16 or values.ndim != 1 or values.numel() != indices.numel():
                raise ValueError("sparse values must be a matching one-dimensional BF16 tensor")
            if not indices.numel() or indices[0] < 0 or indices[-1] >= param.numel():
                raise ValueError("sparse indices are empty or out of range")
            if not torch.all(indices[1:] > indices[:-1]):
                raise ValueError("sparse indices must be strictly increasing")
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite sparse replacement")
            replacement = param.detach().cpu().clone()
            replacement.reshape(-1)[indices] = values
            validate(name, replacement)
        # Transfer all replacements first, so allocation errors cannot partially
        # modify the model during ordinary validation/loading failures.
        prepared = [(param, value.to(param.device)) for param, value in replacements.values()]
        with torch.no_grad():
            for param, value in prepared:
                param.copy_(value)
        self.invalidate_cache()
        return metadata
