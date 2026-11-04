"""Small declarative operations over explicit weight slices and read-only handles."""
import math
import secrets
import torch
from .config import Limits


def selection(shape, specification=None):
    if specification is None:
        return tuple(slice(None) for _ in shape)
    if len(specification) != len(shape):
        raise ValueError("One index or [start, stop] pair is required per dimension")
    result = []
    for size, index in zip(shape, specification):
        if type(index) is int and 0 <= index < size:
            result.append(index)
        elif isinstance(index, list) and len(index) == 2 and all(type(x) is int for x in index) and 0 <= index[0] < index[1] <= size:
            result.append(slice(*index))
        else:
            raise ValueError("Selection is out of bounds or not contiguous")
    return tuple(result)


def finite_scalar(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected a finite scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Scalar exceeds FP32 range") from exc
    if not math.isfinite(result) or abs(result) > torch.finfo(torch.float32).max:
        raise ValueError("Expected a finite FP32 scalar")
    return result


class TensorStore:
    def __init__(self, runtime, limits=None):
        self.runtime = runtime
        self.limits = limits or Limits()
        self.params = runtime.parameters()
        self.aliases = {}
        self.writable = {}
        for tensor in runtime.manifest()["tensors"]:
            name = tensor["name"]
            for alias in tensor["aliases"]:
                self.aliases[alias] = name
            self.writable[name] = tensor["writable"]
        self.originals = {}

    def canonical(self, name):
        if name not in self.aliases:
            raise ValueError("Unknown tensor")
        return self.aliases[name]

    def view(self, name, indices=None):
        name = self.canonical(name)
        value = self.params[name]
        return value[selection(value.shape, indices)]

    def activate(self, replacements):
        with torch.no_grad():
            for name, value in self.originals.items():
                self.params[name].copy_(value)
            for name, value in replacements.items():
                self.params[name].copy_(value)
        self.runtime.invalidate_cache()

    def capture(self, session, tensor, source="weight"):
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if source == "inspection":
            session.counters.charge(self.limits, inspection_scalars=tensor.numel())
        else:
            session.counters.charge(self.limits, weight_scalars=tensor.numel())
        key = secrets.token_hex(16)
        session.handles[key] = tensor
        return {"handle": key, "shape": list(tensor.shape), "scalars": tensor.numel()}

    def read(self, session, tensor=None, selection=None, summary=False, handle=None):
        if tensor is None and handle is None:
            return self.runtime.manifest()
        if (tensor is None) == (handle is None):
            raise ValueError("Select exactly one tensor or handle")
        if handle is not None:
            value = session.handles[handle]
            value = value[globals()["selection"](value.shape, selection)]
        else:
            value = self.view(tensor, selection)
        if summary:
            # Fixed summaries reveal six numbers, independent of tensor shape.
            session.counters.charge(self.limits, weight_scalars=6)
            flat = value.detach().reshape(-1)
            total = squared = 0.0
            lo, hi = float("inf"), -float("inf")
            for chunk in flat.split(1048576):
                f = chunk.double()
                total += f.sum().item()
                squared += f.square().sum().item()
                lo, hi = min(lo, f.min().item()), max(hi, f.max().item())
            return {"shape": list(value.shape), "count": value.numel(), "mean": total/value.numel(), "l2": math.sqrt(squared), "min": lo, "max": hi}
        if value.numel() > self.limits.tensor_elements:
            raise ValueError("Select a smaller weight slice")
        result = self.capture(session, value)
        if value.numel() <= self.limits.returned_scalars:
            result["values"] = value.detach().float().cpu().tolist()
        return result

    def expression(self, session, expr, depth=0, nodes=None):
        nodes = [0] if nodes is None else nodes
        nodes[0] += 1
        if depth > self.limits.expression_depth or nodes[0] > self.limits.expression_nodes:
            raise ValueError("Tensor expression exceeds its complexity limit")
        if isinstance(expr, (int, float)):
            return torch.tensor(finite_scalar(expr), dtype=torch.float32)
        if not isinstance(expr, dict):
            raise ValueError("Expected a declarative tensor expression")
        if "handle" in expr:
            value = session.handles[expr["handle"]]
            value = value[selection(value.shape, expr.get("selection"))]
        elif "tensor" in expr:
            value = self.view(expr["tensor"], expr.get("selection"))
        elif "values" in expr:
            try:
                value = torch.tensor(expr["values"], dtype=torch.float32)
            except (RuntimeError, OverflowError, TypeError, ValueError) as exc:
                raise ValueError("Invalid FP32 literal tensor") from exc
        elif expr.get("op") in {"add", "subtract", "outer", "scale"}:
            op = expr["op"]
            a = self.expression(session, expr["left"], depth+1, nodes)
            if op == "scale":
                value = a * finite_scalar(expr["factor"])
            else:
                b = self.expression(session, expr["right"], depth+1, nodes)
                if op == "outer":
                    if a.ndim != 1 or b.ndim != 1 or a.numel()*b.numel() > self.limits.tensor_elements:
                        raise ValueError("Outer product exceeds bounds")
                    value = torch.outer(a, b)
                else:
                    if a.shape != b.shape:
                        raise ValueError("Linear combinations require equal shapes")
                    value = a+b if op == "add" else a-b
        else:
            raise ValueError("Unknown tensor operation")
        if value.numel() > self.limits.tensor_elements:
            raise ValueError("Expression tensor is too large")
        value = value.detach().float().cpu()
        if not torch.isfinite(value).all():
            raise ValueError("Tensor expression contains a non-finite value")
        return value

    def write(self, session, tensor, operation, selection=None, value=None, source=None, factor=None, left=None, right=None):
        name = self.canonical(tensor)
        if not self.writable[name]:
            raise ValueError("Tensor is frozen")
        parameter = self.params[name]
        view = parameter[globals()["selection"](parameter.shape, selection)]
        if operation == "low_rank_add":
            a, b = self.expression(session, left), self.expression(session, right)
            if view.ndim != 2 or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.shape[1] > 8 or (a.shape[0], b.shape[1]) != view.shape:
                raise ValueError("Low-rank edit dimensions must match, with rank at most eight")
            if view.numel() > 67108864:
                raise ValueError("Low-rank edit exceeds parameter limit")
            replacement = view.detach().float().cpu() + finite_scalar(1.0 if factor is None else factor) * (a @ b)
        else:
            if view.numel() > self.limits.tensor_elements:
                raise ValueError("Select a smaller weight slice")
            current = view.detach().float().cpu()
            if operation == "fill":
                replacement = torch.full_like(current, finite_scalar(value))
            elif operation == "scale":
                replacement = current * finite_scalar(factor)
            elif operation in {"replace", "copy", "add"}:
                replacement = self.expression(session, source if operation == "copy" else value)
                if replacement.shape != current.shape:
                    raise ValueError("Replacement shape must match exactly")
                if operation == "add":
                    replacement = current + replacement
            else:
                raise ValueError("Unknown weight operation")
        replacement = replacement.to(parameter.dtype)
        if not torch.isfinite(replacement).all():
            raise ValueError("Edit overflows the evaluation dtype")
        if name not in self.originals:
            self.originals[name] = parameter.detach().cpu().clone()
        with torch.no_grad():
            view.copy_(replacement)
        session.replacements[name] = parameter.detach().cpu().clone()
        session.revision += 1
        session.handles.clear()
        self.runtime.invalidate_cache()
        return {"tensor": name, "revision": session.revision, "selected_parameters": view.numel()}

    def patch_statistics(self, replacements):
        changed, norm_sq, maximum = 0, 0.0, 0.0
        for name, value in replacements.items():
            for new, old in zip(value.reshape(-1).split(1048576), self.originals[name].reshape(-1).split(1048576)):
                delta = new.double()-old.double()
                changed += int(torch.count_nonzero(delta))
                norm_sq += float(delta.square().sum())
                maximum = max(maximum, float(delta.abs().max()))
        return {"changed_parameters": changed, "l2": math.sqrt(norm_sq), "linf": maximum, "tensors": len(replacements)}
