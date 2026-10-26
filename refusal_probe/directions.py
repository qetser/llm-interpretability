import functools
import json
import math
import os

import torch
from tqdm import tqdm

from .hooks import add_hooks


# ---------------- scoring ----------------
def refusal_score(logits, refusal_toks, eps=1e-8):
    """log-odds of the refusal tokens at the last position."""
    probs = logits[:, -1, :].to(torch.float64).softmax(-1)
    p = probs[:, refusal_toks].sum(-1)
    return torch.log(p + eps) - torch.log(1 - p + eps)


@torch.no_grad()
def last_logits(pm, instructions, pre_hooks=(), hooks=(), batch_size=32):
    out = []
    for i in range(0, len(instructions), batch_size):
        toks = pm.tokenize(instructions[i : i + batch_size])
        with add_hooks(list(pre_hooks), list(hooks)):
            lg = pm.model(input_ids=toks.input_ids.to(pm.device), attention_mask=toks.attention_mask.to(pm.device), logits_to_keep=1).logits
        out.append(lg[:, -1:, :].float().cpu())
    return torch.cat(out)  # [n, 1, vocab]


def refusal_scores(pm, instructions, pre_hooks=(), hooks=(), batch_size=32):
    lg = last_logits(pm, instructions, pre_hooks, hooks, batch_size)
    return refusal_score(lg, pm.refusal_toks).cpu()


def kl_div(logits_a, logits_b, eps=1e-6):
    pa, pb = logits_a.double().softmax(-1), logits_b.double().softmax(-1)
    return (pa * (torch.log(pa + eps) - torch.log(pb + eps))).sum(-1).mean(-1)


def filter_by_refusal(pm, harmful, harmless):
    hs = refusal_scores(pm, harmful)
    ls = refusal_scores(pm, harmless)
    return [x for x, s in zip(harmful, hs.tolist()) if s > 0], [x for x, s in zip(harmless, ls.tolist()) if s < 0]


# ---------------- candidate directions ----------------
@torch.no_grad()
def mean_activations(pm, instructions, positions, batch_size=32):
    cache = torch.zeros((len(positions), pm.n_layers, pm.d_model), dtype=torch.float64, device=pm.device)
    n = len(instructions)

    def make(layer):
        def fn(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            cache[:, layer] += h[:, positions, :].to(cache).sum(0) / n

        return fn

    pre = [(pm.blocks[l], make(l)) for l in range(pm.n_layers)]
    for i in tqdm(range(0, n, batch_size), desc="mean acts", mininterval=30):
        toks = pm.tokenize(instructions[i : i + batch_size])
        with add_hooks(pre, []):
            pm.model.model(input_ids=toks.input_ids.to(pm.device), attention_mask=toks.attention_mask.to(pm.device))  # skip lm_head
    return cache


def generate_directions(pm, harmful, harmless):
    positions = list(range(-len(pm.eoi_toks), 0))
    d = mean_activations(pm, harmful, positions) - mean_activations(pm, harmless, positions)
    assert not d.isnan().any()
    return d  # [n_pos, n_layers, d_model]


# ---------------- selection ----------------
def select_direction(pm, harmful_val, harmless_val, candidates, artifact_dir, kl_threshold=0.1,
                     induce_refusal_threshold=0.0, prune_layer_percentage=0.2):
    os.makedirs(artifact_dir, exist_ok=True)
    n_pos, n_layer, _ = candidates.shape

    base_harmful = refusal_scores(pm, harmful_val).mean().item()
    base_harmless = refusal_scores(pm, harmless_val).mean().item()
    base_logits = last_logits(pm, harmless_val)

    kl = torch.zeros(n_pos, n_layer, dtype=torch.float64)
    abl = torch.zeros_like(kl)
    steer = torch.zeros_like(kl)

    for p in range(-n_pos, 0):
        for l in tqdm(range(n_layer), desc=f"select pos {p}", mininterval=30):
            d = candidates[p, l]
            pre, post = pm.ablation_hooks(d)
            kl[p, l] = kl_div(base_logits, last_logits(pm, harmless_val, pre, post)).mean().item()
            abl[p, l] = refusal_scores(pm, harmful_val, pre, post).mean().item()
            pre, post = pm.addition_hooks(l, d, 1.0)
            steer[p, l] = refusal_scores(pm, harmless_val, pre, post).mean().item()

    plot_scores(abl, base_harmful, pm, "Ablating direction on harmful prompts", artifact_dir, "ablation_scores")
    plot_scores(steer, base_harmless, pm, "Adding direction on harmless prompts", artifact_dir, "actadd_scores")
    plot_scores(kl, 0.0, pm, "KL divergence when ablating on harmless prompts", artifact_dir, "kl_div_scores")

    rows = [dict(position=p, layer=l, refusal_score=abl[p, l].item(), steering_score=steer[p, l].item(), kl_div_score=kl[p, l].item())
            for p in range(-n_pos, 0) for l in range(n_layer)]

    def keep(kl_t, steer_t):
        return sorted((r for r in rows
                       if not any(math.isnan(r[k]) for k in ("refusal_score", "steering_score", "kl_div_score"))
                       and r["layer"] < int(n_layer * (1 - prune_layer_percentage))
                       and r["kl_div_score"] <= kl_t
                       and (steer_t is None or r["steering_score"] >= steer_t)), key=lambda r: r["refusal_score"])

    # Small models often have no direction meeting the paper's thresholds; relax in steps and record it.
    ladder = [(kl_threshold, induce_refusal_threshold), (2.5 * kl_threshold, induce_refusal_threshold),
              (5 * kl_threshold, induce_refusal_threshold), (5 * kl_threshold, None)]
    for step, (kl_t, st_t) in enumerate(ladder):
        kept = keep(kl_t, st_t)
        if kept:
            break
    else:
        raise RuntimeError("All candidate directions were filtered out even after relaxing thresholds")
    relaxed = None if step == 0 else dict(kl_threshold=kl_t, induce_refusal_threshold=st_t)
    if relaxed:
        print(f"WARNING: no direction met kl<={kl_threshold}, steer>={induce_refusal_threshold}; relaxed to {relaxed}")
    json.dump(rows, open(f"{artifact_dir}/direction_evaluations.json", "w"), indent=2)
    json.dump(kept, open(f"{artifact_dir}/direction_evaluations_filtered.json", "w"), indent=2)
    best = kept[0]
    print(f"Selected pos={best['position']} layer={best['layer']} | ablated refusal {best['refusal_score']:.3f} "
          f"(base {base_harmful:.3f}) | steer {best['steering_score']:.3f} (base {base_harmless:.3f}) | KL {best['kl_div_score']:.4f}")
    return best["position"], best["layer"], candidates[best["position"], best["layer"]], dict(
        baseline_harmful_refusal=base_harmful, baseline_harmless_refusal=base_harmless, relaxed_thresholds=relaxed, **best)


def plot_scores(scores, baseline, pm, title, artifact_dir, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pos, n_layer = scores.shape
    labels = pm.tokenizer.batch_decode([[t] for t in pm.eoi_toks])
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(-n_pos, 0):
        ax.plot(range(n_layer), scores[i].numpy(), label=f"{i}: {labels[i]!r}")
    ax.axhline(baseline, color="black", linestyle="--")
    ax.set(title=title, xlabel="Layer source of direction (resid_pre)", ylabel="Refusal score")
    ax.legend(title="Position source", loc="lower left")
    fig.savefig(f"{artifact_dir}/{name}.png", dpi=110)
    plt.close(fig)
