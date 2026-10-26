"""Defense 1: fine-tune on refusals with diverse first tokens (Labunets 2026, arXiv 2608.25390).

A single repeated refusal opener makes the refusal residual low-rank, so one difference-in-means direction carries it.
--m is the number of distinct openers; m=1 is the control with the same data and step count.
"""
import argparse
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..bench.safety import is_refusal
from ..data import load_split

OPENERS = [
    "I'm sorry, but I can't help with that.",
    "Sorry, I can't assist with that request.",
    "Unfortunately, I'm not able to help with that.",
    "No, I won't help with that.",
    "Apologies, but I cannot assist with this.",
    "Regrettably, I can't provide that.",
    "Sadly, I am unable to help with this request.",
    "My apologies, but I can't help with that.",
    "Thanks for asking, but I must decline.",
    "Absolutely not, I can't help with that.",
    "That's something I can't help with.",
    "Hmm, I can't assist with that one.",
    "Respectfully, I must decline to help with that.",
    "Afraid I can't help with that request.",
    "Pardon me, but I can't help with this.",
    "Alas, I am unable to assist with that.",
]
TAIL = " If there's something else I can help with, just ask."
assert all(is_refusal(o) for o in OPENERS)


def _prompt_ids(tok, instruction):
    text = tok.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def self_completions(model, tok, prompts, max_new=96, bs=32):
    """The model's own answers to harmless prompts, kept as retention data. EOS only if the answer ended naturally."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for g in gen[:, enc.input_ids.shape[-1]:].tolist():
            if tok.eos_token_id in g:
                g = g[:g.index(tok.eos_token_id) + 1]
            else:
                g = [t for t in g if t != tok.pad_token_id]
            out.append(g)
    return out


def build(model, tok, m, n_harmful, n_harmless, seed):
    rng = random.Random(seed)
    harmful = rng.sample(load_split("harmful", "train", True), n_harmful)
    harmless = rng.sample(load_split("harmless", "train", True), n_harmless)
    ex = []
    for i, p in enumerate(harmful):
        resp = tok(OPENERS[i % m] + TAIL, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ex.append((_prompt_ids(tok, p), resp))
    for p, r in zip(harmless, self_completions(model, tok, harmless)):
        ex.append((_prompt_ids(tok, p), r))
    rng.shuffle(ex)
    return ex


def batches(ex, bs, pad):
    for i in range(0, len(ex), bs):
        chunk = ex[i:i + bs]
        L = max(len(p) + len(r) for p, r in chunk)
        ids = torch.full((len(chunk), L), pad)
        lab = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (p, r) in enumerate(chunk):
            n = len(p) + len(r)
            ids[j, :n] = torch.tensor(p + r)
            lab[j, len(p):n] = torch.tensor(r)
            att[j, :n] = 1
        yield ids, att, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--m", type=int, required=True, help="number of distinct refusal openers (1..16)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-harmful", type=int, default=200)
    ap.add_argument("--n-harmless", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert 1 <= a.m <= len(OPENERS)

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda().eval()
    ex = build(model, tok, a.m, a.n_harmful, a.n_harmless, a.seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(ex) + a.bs - 1) // a.bs)
    step = 0
    for ep in range(a.epochs):
        random.Random(a.seed + ep).shuffle(ex)
        for ids, att, lab in batches(ex, a.bs, tok.pad_token_id):
            loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
