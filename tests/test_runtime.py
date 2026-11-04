"""Native tiny-config CPU tests. These do not establish benchmark edit success."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

from iwibench.runtime import NativeTarget, PATCH_FORMAT, alias_tokenization


@pytest.fixture
def target():
    config = Gemma3TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, sliding_window=16, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, attn_implementation="eager",
    )
    model = Gemma3ForCausalLM(config).to(torch.bfloat16)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,
                                  "[UNK]": 3, "hello": 4, "world": 5,
                                  "user": 6, "model": 7}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{{ bos_token }} user {{ messages[0]['content'] }} model"
    instance = NativeTarget.__new__(NativeTarget)
    instance._bind(model, tokenizer, "google/gemma-3-1b-it", "a" * 40)
    return instance


def metadata(target):
    return {"format": PATCH_FORMAT, "model_id": target.model_id, "revision": target.revision}


def test_native_aliases_and_generation_counts(target):
    params = target.parameters()
    assert params["model.embed_tokens.weight"] is params["lm_head.weight"]
    manifest = target.manifest()
    group = next(x for x in manifest["tensors"] if "lm_head.weight" in x["aliases"])
    assert group["aliases"] == ["model.embed_tokens.weight", "lm_head.weight"]
    batch = target.generate(["hello world", "hello"], 3, [7, 8])
    result = batch[0]
    assert result["input_tokens"] == target.count_tokens("hello world")
    assert 1 <= result["output_tokens"] <= 3
    assert isinstance(result["text"], str)
    assert batch[1]["input_tokens"] == target.count_tokens("hello")
    assert batch[0]["input_tokens"] > batch[1]["input_tokens"]


def test_inspection_is_read_only_and_hooks_are_removed(target):
    before = {name: p.clone() for name, p in target.parameters().items()}
    result = target.inspect(["hello world"], [0, 1], [0, -1],
                            ["residual", "attention", "mlp", "statistics", "probabilities"], [4, 5])
    sequence = result["sequences"][0]
    assert result["input_tokens"] == 2
    assert sequence["probabilities"].shape == (2, 2)
    assert sequence["activations"]["0"]["residual"].shape == (2, 16)
    assert sequence["activations"]["1"]["statistics"].shape == (2, 5)
    assert not any(module._forward_hooks for module in target.model.modules())
    assert all(torch.equal(before[name], p) for name, p in target.parameters().items())
    assert all(not p.requires_grad for p in target.parameters().values())


def test_hooks_removed_when_forward_raises(target, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("planted forward failure")
    monkeypatch.setattr(target.model, "forward", fail)
    with pytest.raises(RuntimeError, match="planted"):
        target.inspect(["hello"], [0], [0], ["residual"], [])
    assert not any(module._forward_hooks for module in target.model.modules())


def test_patch_round_trip_updates_both_tied_names(target, tmp_path):
    replacement = target.parameters()["lm_head.weight"].detach().clone()
    replacement[4].fill_(0.5)
    path = tmp_path / "patch.safetensors"
    save_file({"lm_head.weight": replacement}, str(path), metadata=metadata(target))
    target.apply_patch(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert torch.equal(target.parameters()["model.embed_tokens.weight"], replacement)
    assert target.parameters()["lm_head.weight"] is target.parameters()["model.embed_tokens.weight"]
    assert target.weight_revision == 1


def test_patch_digest_mismatch_does_not_mutate(target, tmp_path):
    before = target.parameters()["lm_head.weight"].detach().clone()
    path = tmp_path / "patch.safetensors"
    save_file({"lm_head.weight": before + 1}, str(path), metadata=metadata(target))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        target.apply_patch(path, expected_sha256="0" * 64)
    assert torch.equal(before, target.parameters()["lm_head.weight"])
    assert target.weight_revision == 0


def test_grading_worker_requires_patch_digest_before_loading_model(monkeypatch):
    from iwibench import grading_worker

    def fail(*args, **kwargs):
        pytest.fail("model loaded before patch digest validation")

    monkeypatch.setattr(grading_worker, "NativeTarget", fail)
    with pytest.raises(ValueError, match="patch_sha256 is required"):
        grading_worker.run({"patch_path": "patch.safetensors"})


def test_grading_worker_passes_digest_to_patch_loader(target, tmp_path, monkeypatch):
    from iwibench import grading_worker

    before = target.parameters()["lm_head.weight"].detach().clone()
    path = tmp_path / "patch.safetensors"
    save_file({"lm_head.weight": before + 1}, str(path), metadata=metadata(target))
    monkeypatch.setattr(grading_worker, "NativeTarget", lambda *args, **kwargs: target)
    request = {"model_id": target.model_id, "revision": target.revision,
               "patch_path": str(path), "patch_sha256": "0" * 64,
               "prompts": ["hello"], "seeds": [0], "max_new_tokens": 1}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        grading_worker.run(request)
    assert torch.equal(before, target.parameters()["lm_head.weight"])
    request["patch_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    grading_worker.run(request)
    assert torch.equal(before + 1, target.parameters()["lm_head.weight"])


def test_conflicting_alias_patch_cannot_partially_apply(target, tmp_path):
    before = target.parameters()["lm_head.weight"].detach().clone()
    path = tmp_path / "conflict.safetensors"
    save_file({"lm_head.weight": before + 1, "model.embed_tokens.weight": before + 2},
              str(path), metadata=metadata(target))
    with pytest.raises(ValueError, match="conflicting"):
        target.apply_patch(path)
    assert torch.equal(before, target.parameters()["lm_head.weight"])
    assert target.weight_revision == 0


def test_frozen_parameter_and_revision_rejected(target, tmp_path):
    target.model.vision_tower = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
    path = tmp_path / "frozen.safetensors"
    save_file({"vision_tower.weight": torch.zeros((16, 16), dtype=torch.bfloat16)},
              str(path), metadata=metadata(target))
    with pytest.raises(ValueError, match="frozen"):
        target.apply_patch(path)
    bad = metadata(target) | {"revision": "b" * 40}
    save_file({"lm_head.weight": target.parameters()["lm_head.weight"].detach().clone()},
              str(path), metadata=bad)
    with pytest.raises(ValueError, match="identity"):
        target.apply_patch(path)


def test_invalid_inspection_positions_do_not_execute(target):
    with pytest.raises(ValueError, match="outside"):
        target.inspect(["hello"], [0], [1], ["residual"], [])
    assert not any(module._forward_hooks for module in target.model.modules())


def test_sparse_patch_preserves_other_values_and_tied_aliases(target, tmp_path):
    before = target.parameters()["lm_head.weight"].detach().clone()
    indices = torch.tensor([0, 20, before.numel() - 1], dtype=torch.int64)
    values = torch.tensor([0.25, 0.5, 0.75], dtype=torch.bfloat16)
    expected = before.clone()
    expected.reshape(-1)[indices] = values
    path = tmp_path / "sparse.safetensors"
    sparse = [{"name": "lm_head.weight", "indices": "__iwi_indices_0", "values": "__iwi_values_0"}]
    save_file({"__iwi_indices_0": indices, "__iwi_values_0": values}, str(path),
              metadata=metadata(target) | {"sparse": json.dumps(sparse)})
    target.apply_patch(path)
    assert torch.equal(target.parameters()["lm_head.weight"], expected)
    assert target.parameters()["lm_head.weight"] is target.parameters()["model.embed_tokens.weight"]


@pytest.mark.parametrize("indices", [[2, 2], [3, 1], [-1, 2], [0, 512]])
def test_bad_sparse_indices_do_not_mutate(target, tmp_path, indices):
    before = target.parameters()["lm_head.weight"].detach().clone()
    path = tmp_path / "bad-sparse.safetensors"
    sparse = [{"name": "lm_head.weight", "indices": "__iwi_indices_0", "values": "__iwi_values_0"}]
    save_file({"__iwi_indices_0": torch.tensor(indices, dtype=torch.int64),
               "__iwi_values_0": torch.ones(2, dtype=torch.bfloat16)}, str(path),
              metadata=metadata(target) | {"sparse": json.dumps(sparse)})
    with pytest.raises(ValueError, match="sparse indices"):
        target.apply_patch(path)
    assert torch.equal(before, target.parameters()["lm_head.weight"])


def test_alias_tokenization_uses_actual_chat_occurrences(target):
    ep = {"family": "alias_acquisition", "parameters": {"bindings": [{"alias": "hello"}]},
          "targets": [{"id": "target:0", "prompt": "hello world"}],
          "public_practice": [{"id": "practice:0", "prompt": "hello hello"}]}
    result = alias_tokenization(ep, target)["hello"]
    assert result["token_ids"] == [4]
    assert result["context_token_ids"] == [[4], [4], [4]]
    assert len(result["occurrences"]) == 3
    assert all(x["positions"] for x in result["occurrences"])


def test_grading_worker_batches_native_generation_at_eight(target, monkeypatch):
    from iwibench import grading_worker

    batches = []
    original_generate = target.generate

    def record_generate(prompts, max_new_tokens, seeds):
        batches.append((len(prompts), list(seeds)))
        return original_generate(prompts, max_new_tokens, seeds)

    monkeypatch.setattr(target, "generate", record_generate)
    monkeypatch.setattr(grading_worker, "NativeTarget", lambda *args, **kwargs: target)
    result = grading_worker.run({"model_id": target.model_id, "revision": target.revision,
                                "prompts": ["hello"] * 9, "seeds": list(range(9)),
                                "max_new_tokens": 2})
    assert batches == [(8, list(range(8))), (1, [8])]
    assert len(result["generations"]) == 9
    assert all(1 <= row["output_tokens"] <= 2 for row in result["generations"])
