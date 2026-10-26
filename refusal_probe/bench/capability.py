"""Standard capability benchmarks (lm-evaluation-harness) with and without the refusal direction ablated."""
import json
import os

from ..hooks import add_hooks


def run_lm_eval(pm, tasks, pre_hooks=(), hooks=(), limit=None, batch_size=8, num_fewshot=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=pm.model, tokenizer=pm.tokenizer, batch_size=batch_size)
    with add_hooks(list(pre_hooks), list(hooks)):
        res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit, num_fewshot=num_fewshot, log_samples=False)
    return res["results"]


def headline(results: dict) -> dict:
    """One number per task: acc_norm if present else acc / mc2 / exact_match."""
    out = {}
    for task, r in results.items():
        for key in ("acc_norm,none", "acc,none", "mc2,none", "exact_match,strict-match", "exact_match,flexible-extract"):
            if key in r:
                out[task] = r[key]
                break
    return out
