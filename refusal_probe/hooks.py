import contextlib
import torch
from torch import Tensor


def _unit(direction: Tensor) -> Tensor:
    return direction / (direction.norm() + 1e-8)


def _ablate(act: Tensor, unit: Tensor) -> Tensor:
    u = unit.to(act)
    return act - (act @ u).unsqueeze(-1) * u


def ablation_pre_hook(direction: Tensor):
    unit = _unit(direction.float())

    def fn(module, args, kwargs):
        # decoder blocks receive hidden_states positionally or as a kwarg depending on version
        if args:
            return (_ablate(args[0], unit), *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = _ablate(kwargs["hidden_states"], unit)
        return args, kwargs

    return fn


def ablation_output_hook(direction: Tensor):
    unit = _unit(direction.float())

    def fn(module, args, output):
        if isinstance(output, tuple):
            return (_ablate(output[0], unit), *output[1:])
        return _ablate(output, unit)

    return fn


def addition_pre_hook(vector: Tensor, coeff: float):
    def fn(module, args, kwargs):
        if args:
            return (args[0] + coeff * vector.to(args[0]), *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = kwargs["hidden_states"] + coeff * vector.to(kwargs["hidden_states"])
        return args, kwargs

    return fn


@contextlib.contextmanager
def add_hooks(pre_hooks, hooks):
    """Pre-hooks are registered with_kwargs=True (see the *_pre_hook signatures)."""
    handles = []
    try:
        for module, fn in pre_hooks:
            handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
        for module, fn in hooks:
            handles.append(module.register_forward_hook(fn))
        yield
    finally:
        for h in handles:
            h.remove()


def build_ablation_hooks(blocks, attns, mlps, direction: Tensor):
    pre = [(b, ablation_pre_hook(direction)) for b in blocks]
    post = [(m, ablation_output_hook(direction)) for m in attns] + [(m, ablation_output_hook(direction)) for m in mlps]
    return pre, post


def build_addition_hooks(blocks, layer: int, vector: Tensor, coeff: float):
    return [(blocks[layer], addition_pre_hook(vector, coeff))], []
