"""Cross-entropy of the model on its own harmless completions (paper's 'on-distribution CE')."""
import torch

from ..hooks import add_hooks


@torch.no_grad()
def ce_loss(pm, prompts, responses, pre_hooks=(), hooks=(), batch_size=1):
    tok = pm.tokenizer
    total, count = 0.0, 0
    for i in range(0, len(prompts), batch_size):
        P = [pm.format_prompt(p) for p in prompts[i : i + batch_size]]
        R = responses[i : i + batch_size]
        ids, masks = [], []
        for p, r in zip(P, R):
            pi = tok.encode(p, add_special_tokens=False)
            ri = tok.encode(r, add_special_tokens=False)
            ids.append(pi + ri)
            masks.append([0] * len(pi) + [1] * len(ri))
        L = max(map(len, ids))
        pad = tok.pad_token_id
        x = torch.tensor([[pad] * (L - len(s)) + s for s in ids])
        m = torch.tensor([[0] * (L - len(s)) + s for s in masks])
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ids])
        with add_hooks(list(pre_hooks), list(hooks)):
            logits = pm.model(input_ids=x.to(pm.device), attention_mask=att.to(pm.device)).logits
        tgt = x[:, 1:].to(pm.device)
        mk = m[:, 1:].to(pm.device).bool()
        sel = logits[:, :-1][mk].float()  # response positions only keeps the fp32 copy small
        total += torch.nn.functional.cross_entropy(sel, tgt[mk], reduction="sum").item()
        count += mk.sum().item()
    return total / max(count, 1)
