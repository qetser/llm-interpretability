from collections import Counter, defaultdict
from typing import List

from .data import load_split


def detect_refusal_tokens(pm, n_prompts=48, top_k=4, min_precision=0.9, min_count=3) -> List[int]:
    """Model-agnostic replacement for the paper's hand-picked refusal tokens.

    Greedy-decode a short completion for harmful and harmless prompts, label each as refusal/non-refusal with
    the substring matcher, and keep first tokens that (almost) always start a refusal.
    """
    from .bench.safety import is_refusal

    prompts = load_split("harmful", "train", True)[:n_prompts] + load_split("harmless", "train", True)[:n_prompts]
    toks = defaultdict(lambda: [0, 0])  # first token -> [n_refusal, n_total]
    for i in range(0, len(prompts), 16):
        batch = prompts[i : i + 16]
        enc = pm.tokenize(batch)
        import torch

        with torch.no_grad():
            gen = pm.model.generate(input_ids=enc.input_ids.to(pm.device), attention_mask=enc.attention_mask.to(pm.device),
                                    max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None,
                                    pad_token_id=pm.tokenizer.pad_token_id)
        gen = gen[:, enc.input_ids.shape[-1]:]
        for g in gen:
            text = pm.tokenizer.decode(g, skip_special_tokens=True)
            t = g[0].item()
            toks[t][1] += 1
            toks[t][0] += int(is_refusal(text))
    cands = [(r, t) for t, (r, n) in toks.items() if r >= min_count and r / n >= min_precision]
    cands.sort(reverse=True)
    out = [t for _, t in cands[:top_k]]
    if not out:
        raise RuntimeError("could not detect refusal tokens; model may not refuse on this prompt set")
    return out
