"""Reservations occur under the service lock before any GPU work."""
from dataclasses import dataclass, asdict
from .config import Limits


class BudgetExceeded(ValueError):
    pass


@dataclass
class Counters:
    tool_calls: int = 0
    run_calls: int = 0
    run_sequences: int = 0
    inspect_calls: int = 0
    inspect_sequences: int = 0
    forward_tokens: int = 0
    inspection_scalars: int = 0
    weight_scalars: int = 0

    def charge(self, limits: Limits, **increments):
        for key, amount in increments.items():
            if not isinstance(amount, int) or amount < 0:
                raise ValueError("Charges must be nonnegative integers")
            if getattr(self, key) + amount > getattr(limits, key):
                raise BudgetExceeded(f"{key} limit exhausted")
        for key, amount in increments.items():
            setattr(self, key, getattr(self, key) + amount)

    def refund_generation(self, reserved: int, used: int):
        if not 0 <= used <= reserved:
            raise ValueError("Invalid generation accounting")
        self.forward_tokens -= reserved - used

    def as_dict(self):
        return asdict(self)
