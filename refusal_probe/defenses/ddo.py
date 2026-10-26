"""Defense 2: Decoy Direction Optimization, simplified (Muhamed et al. 2026, arXiv 2609.16204).

A few low-norm MLP neurons per layer read the refusal coordinate of the MLP input and write a learned decoy vector
orthogonal to the residual refusal direction. Difference-in-means then picks up the decoy instead of the refusal feature.
Differences from the paper: one reader group, replace mode only, and the self-RFA confusion loss ablates a direction
re-estimated from the current batch at a random layer.
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch import nn

from ..data import load_split
from ..directions import filter_by_refusal, generate_directions, kl_div, refusal_score
from ..hooks import add_hooks
from ..model import ProbedModel
from ..refusal_tokens import detect_refusal_tokens


class DecoyMLP(nn.Module):
    def __init__(self, mlp, idx, reader, resid_dir, amp0):
        super().__init__()
        self.mlp = mlp
        k, d = len(idx), reader.numel()
        mask = torch.ones(mlp.down_proj.in_features)
        mask[idx] = 0
        self.register_buffer("mask", mask)
        self.register_buffer("reader", reader / reader.norm())
        self.register_buffer("resid_unit", resid_dir / resid_dir.norm())
        self.idx = idx
        self.dec = nn.Parameter(torch.randn(k, d) / d ** 0.5)
        self.log_amp = nn.Parameter(torch.full((k,), float(amp0)).log())
        self.log_s = nn.Parameter(torch.zeros(k))

    def decoys(self):
        d = self.dec - (self.dec @ self.resid_unit)[:, None] * self.resid_unit
        return F.normalize(d, dim=-1)

    def forward(self, x):
        m = self.mlp
        out = m.down_proj(m.act_fn(m.gate_proj(x)) * m.up_proj(x) * self.mask.to(x))
        c = (x @ self.reader.to(x))[..., None]
        g = F.silu(c * self.log_s.exp()) * c
        return out + (g * self.log_amp.exp()) @ self.decoys()

    @torch.no_grad()
    def bake(self):
        m, i = self.mlp, self.idx
        m.gate_proj.weight[i] = self.log_s.exp()[:, None] * self.reader[None]
        m.up_proj.weight[i] = self.reader[None].expand(len(i), -1)
        m.down_proj.weight[:, i] = (self.log_amp.exp()[:, None] * self.decoys()).T
        return m


def mlp_input_means(pm, prompts, layers, bs=32):
    """Mean of the normalised MLP input at the last position, per layer."""
    acc = {l: 0 for l in layers}
    cur = {}
    pre = [(pm.mlps[l], (lambda l: lambda mod, args, kw: cur.__setitem__(l, (args[0] if args else kw["x"])[:, -1].double()))(l)) for l in layers]
    for i in range(0, len(prompts), bs):
        toks = pm.tokenize(prompts[i:i + bs])
        with torch.no_grad(), add_hooks(pre, []):
            pm.model.model(input_ids=toks.input_ids.to(pm.device), attention_mask=toks.attention_mask.to(pm.device))
        for l in layers:
            acc[l] = acc[l] + cur[l].sum(0)
    return {l: (acc[l] / len(prompts)).float() for l in layers}


def forward_capture(pm, prompts, layer, pre_hooks=(), hooks=()):
    """Last-position logits plus resid_pre of `layer` at the last position (keeps the autograd graph)."""
    store = {}

    def grab(mod, args, kwargs):
        store["h"] = (args[0] if args else kwargs["hidden_states"])[:, -1]

    toks = pm.tokenize(prompts)
    with add_hooks([(pm.blocks[layer], grab)] + list(pre_hooks), list(hooks)):
        lg = pm.model(input_ids=toks.input_ids.to(pm.device), attention_mask=toks.attention_mask.to(pm.device), logits_to_keep=1).logits[:, -1]
    return lg, store["h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(6, 16)))
    ap.add_argument("--attack-layers", type=int, nargs="+", default=[12, 14, 16, 17])
    ap.add_argument("--k", type=int, default=8, help="decoy neurons per layer")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--w-ret", type=float, default=1.0)
    ap.add_argument("--w-conf", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    pm = ProbedModel(a.model, dtype=torch.float32, device_map={"": 0 if torch.cuda.is_available() else "cpu"})
    pm.refusal_toks = detect_refusal_tokens(pm)
    harmful = rng.sample(load_split("harmful", "train", True), 200)
    harmless = rng.sample(load_split("harmless", "train", True), 200)
    harmful, harmless = filter_by_refusal(pm, harmful, harmless)
    print(f"{len(harmful)} harmful / {len(harmless)} harmless prompts after refusal filtering")

    r_res = generate_directions(pm, harmful[:128], harmless[:128])[-1]  # last position, [n_layers, d]
    r_in = mlp_input_means(pm, harmful[:128], a.layers)
    r_in_h = mlp_input_means(pm, harmless[:128], a.layers)

    wrapped = {}
    for l in a.layers:
        mlp = pm.mlps[l]
        idx = mlp.down_proj.weight.norm(dim=0).argsort()[: a.k]
        reader = (r_in[l] - r_in_h[l]).to(pm.device)
        amp0 = 2.0 * r_res[l].norm().item() / (a.k * max(reader.norm().item(), 1e-3) ** 2)
        wrapped[l] = DecoyMLP(mlp, idx, reader, r_res[l].float(), amp0).to(pm.device)
    with torch.no_grad():
        base_logits = torch.cat([
            pm.model(**{k: v.to(pm.device) for k, v in pm.tokenize(harmless[i:i + 32]).items()}, logits_to_keep=1).logits[:, -1]
            for i in range(0, len(harmless), 32)])
    for l, w in wrapped.items():
        pm.blocks[l].mlp = w
        pm.mlps[l] = w  # so ablation hooks see the decoy output too
    params = [p for w in wrapped.values() for p in (w.dec, w.log_amp, w.log_s)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=a.lr)

    for step in range(1, a.steps + 1):
        h = rng.sample(harmful, a.bs)
        j = rng.randrange(0, len(harmless) - a.bs)
        s = harmless[j:j + a.bs]
        La = rng.choice(a.attack_layers)

        lg_h, act_h = forward_capture(pm, h, La)
        lg_s, act_s = forward_capture(pm, s, La)
        l_ref = F.relu(a.margin - refusal_score(lg_h[:, None], pm.refusal_toks)).mean()
        l_ret = kl_div(base_logits[j:j + a.bs], lg_s).mean()

        r_est = act_h.mean(0) - act_s.mean(0)
        pre, post = pm.ablation_hooks(r_est)
        lg_a, _ = forward_capture(pm, h, La, pre, post)
        l_conf = kl_div(lg_h.detach(), lg_a).mean() + F.relu(a.margin - refusal_score(lg_a[:, None], pm.refusal_toks)).mean()

        loss = l_ref + a.w_ret * l_ret + a.w_conf * l_conf
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f} ref {l_ref.item():.3f} ret {l_ret.item():.4f} conf {l_conf.item():.3f}", flush=True)

    # fold the decoys into plain weights and check nothing changed
    probe = harmful[:8] + harmless[:8]
    with torch.no_grad():
        before = pm.model(**{k: v.to(pm.device) for k, v in pm.tokenize(probe).items()}, logits_to_keep=1).logits[:, -1]
        for l, w in wrapped.items():
            pm.blocks[l].mlp = pm.mlps[l] = w.bake()
        after = pm.model(**{k: v.to(pm.device) for k, v in pm.tokenize(probe).items()}, logits_to_keep=1).logits[:, -1]
    print("bake max logit diff:", (before - after).abs().max().item())
    os.makedirs(a.out, exist_ok=True)
    pm.model.to(torch.bfloat16).save_pretrained(a.out)
    pm.tokenizer.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
