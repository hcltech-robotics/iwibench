"""Frozen target identities and initial development limits."""
from dataclasses import dataclass, asdict

MODELS = {
    "google/gemma-3-1b-it": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}
VERIFIERS_REVISION = "5dad16d40e2d025672fdb7aa75b6540c0053e4e0"


@dataclass(frozen=True)
class Limits:
    tool_calls: int = 128
    run_calls: int = 8
    run_sequences: int = 64
    inspect_calls: int = 4
    inspect_sequences: int = 16
    forward_tokens: int = 262144
    input_tokens: int = 2048
    output_tokens: int = 256
    inspection_positions: int = 16
    inspection_scalars: int = 262144
    returned_scalars: int = 4096
    weight_scalars: int = 1048576
    tensor_elements: int = 1048576
    expression_nodes: int = 32
    expression_depth: int = 8

    def as_dict(self):
        return asdict(self)
