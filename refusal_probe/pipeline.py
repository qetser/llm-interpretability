import argparse
import json
import os
import random

import torch

from .bench import safety
from .bench.loss import ce_loss
from .config import Config
from .data import load_split
from .directions import filter_by_refusal, generate_directions, select_direction
from .model import ProbedModel
from .refusal_tokens import detect_refusal_tokens


def _mk(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_probe(cfg: Config) -> ProbedModel:
    pm = ProbedModel(cfg.model_path)
    meta_path = os.path.join(cfg.artifact_path(), "direction_metadata.json")
    if os.path.exists(meta_path):
        pm.refusal_toks = json.load(open(meta_path))["refusal_toks"]
    return pm


def extract(cfg: Config, pm: ProbedModel):
    """Steps 1-2: refusal tokens, candidate directions, direction selection."""
    art = _mk(cfg.artifact_path())
    pm.refusal_toks = detect_refusal_tokens(pm)
    print("refusal tokens:", pm.tokenizer.convert_ids_to_tokens(pm.refusal_toks))

    random.seed(42)
    s = lambda h, sp, n: random.sample(load_split(h, sp, True), n)
    h_tr, hl_tr = s("harmful", "train", cfg.n_train), s("harmless", "train", cfg.n_train)
    h_va, hl_va = s("harmful", "val", cfg.n_val), s("harmless", "val", cfg.n_val)
    if cfg.filter_train:
        h_tr, hl_tr = filter_by_refusal(pm, h_tr, hl_tr)
    if cfg.filter_val:
        h_va, hl_va = filter_by_refusal(pm, h_va, hl_va)
    print(f"after filtering: train {len(h_tr)}/{len(hl_tr)}  val {len(h_va)}/{len(hl_va)}")
    assert min(len(h_tr), len(hl_tr), len(h_va), len(hl_va)) >= 8, "too few prompts survived refusal filtering"

    cands = generate_directions(pm, h_tr, hl_tr)
    torch.save(cands, f"{_mk(art + '/generate_directions')}/mean_diffs.pt")
    pos, layer, direction, scores = select_direction(
        pm, h_va, hl_va, cands, _mk(art + "/select_direction"), cfg.kl_threshold, cfg.induce_refusal_threshold, cfg.prune_layer_percentage)

    torch.save(direction.float().cpu(), f"{art}/direction.pt")
    json.dump(dict(model=cfg.model_path, pos=pos, layer=layer, refusal_toks=pm.refusal_toks,
                   n_layers=pm.n_layers, d_model=pm.d_model, direction_norm=direction.norm().item(), scores=scores),
              open(f"{art}/direction_metadata.json", "w"), indent=2)
    return pos, layer, direction


def load_direction(cfg: Config):
    art = cfg.artifact_path()
    meta = json.load(open(f"{art}/direction_metadata.json"))
    return meta["layer"], torch.load(f"{art}/direction.pt")


def standard_conditions(pm, layer, direction):
    return {
        "baseline": ([], []),
        "ablation": pm.ablation_hooks(direction),
        "actadd_remove": pm.addition_hooks(layer, direction, -1.0),
    }


def run_safety(cfg: Config, pm: ProbedModel, judge=None):
    layer, direction = load_direction(cfg)
    out = _mk(cfg.artifact_path() + "/safety")
    datasets = {n: safety.sample(n, cfg.n_eval_per_dataset) for n in cfg.harmful_eval_datasets}
    res = safety.eval_conditions(pm, standard_conditions(pm, layer, direction), datasets, out, cfg.max_new_tokens, cfg.gen_batch_size, judge)

    # harmless side: over-refusal at baseline, and refusal induced by adding the direction
    harmless = safety.sample_harmless_test(cfg.n_test)
    conds = {"baseline": ([], []), "actadd_induce": pm.addition_hooks(layer, direction, 1.0)}
    res["harmless"] = safety.eval_conditions(pm, conds, {"harmless": harmless}, out, cfg.max_new_tokens, cfg.gen_batch_size)["harmless"]
    json.dump(res, open(out + "/summary.json", "w"), indent=2)
    return res


def run_loss(cfg: Config, pm: ProbedModel, n=100):
    layer, direction = load_direction(cfg)
    rows = json.load(open(f"{cfg.artifact_path()}/safety/harmless_baseline.json"))["completions"][:n]
    P, R = [r["prompt"] for r in rows], [r["response"] for r in rows]
    res = {name: ce_loss(pm, P, R, pre, post) for name, (pre, post) in standard_conditions(pm, layer, direction).items()}
    json.dump(res, open(_mk(cfg.artifact_path() + "/loss_evals") + "/ce_loss.json", "w"), indent=2)
    print("CE loss:", res)
    return res


def run_capability(cfg: Config, pm: ProbedModel):
    from .bench.capability import headline, run_lm_eval

    layer, direction = load_direction(cfg)
    out = {}
    for name in ("baseline", "ablation"):
        pre, post = standard_conditions(pm, layer, direction)[name]
        full = run_lm_eval(pm, cfg.lm_eval_tasks, pre, post, limit=cfg.lm_eval_limit, batch_size=cfg.lm_eval_batch_size)
        out[name] = dict(headline=headline(full), full=full)
        print(name, out[name]["headline"])
    json.dump(out, open(_mk(cfg.artifact_path() + "/capability") + "/lm_eval.json", "w"), indent=2, default=str)
    return out


STAGES = ["extract", "safety", "loss", "capability"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--judge", action="store_true", help="also score harmful compliance with the HarmBench Mistral classifier")
    ap.add_argument("--lm-eval-limit", type=int, default=None, help="cap examples per lm-eval task")
    ap.add_argument("--lm-eval-batch-size", type=int, default=8)
    ap.add_argument("--lm-eval-tasks", nargs="+", default=None)
    ap.add_argument("--n-eval", type=int, default=100, help="prompts per harmful eval dataset")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="run even if another run holds the lock or results already exist")
    a = ap.parse_args()

    cfg = Config(model_path=a.model, lm_eval_limit=a.lm_eval_limit, lm_eval_batch_size=a.lm_eval_batch_size, n_eval_per_dataset=a.n_eval, max_new_tokens=a.max_new_tokens)
    if a.lm_eval_tasks:
        cfg.lm_eval_tasks = tuple(a.lm_eval_tasks)
    # several queues can share the box: skip a model another process is already probing, or one that is finished
    art = _mk(cfg.artifact_path())
    lock = os.path.join(art, ".lock")
    if not a.force:
        if os.path.exists(f"{art}/capability/lm_eval.json"):
            print(f"{cfg.model_alias}: already benchmarked, skipping (use --force)")
            return
        try:
            if os.path.exists(lock) and os.path.exists(f"/proc/{int(open(lock).read())}"):
                print(f"{cfg.model_alias}: being probed by pid {open(lock).read()}, skipping")
                return
        except ValueError:
            pass
    open(lock, "w").write(str(os.getpid()))
    try:
        run_stages(a, cfg)
    finally:
        os.path.exists(lock) and os.remove(lock)


def run_stages(a, cfg):
    pm = load_probe(cfg)
    if "extract" in a.stages:
        extract(cfg, pm)
    if "safety" in a.stages:
        judge = None
        if a.judge:
            from .bench.judge import HarmBenchJudge

            judge = HarmBenchJudge()  # needs ~14GB on top of the model being probed
        run_safety(cfg, pm, judge)
    if "loss" in a.stages:
        run_loss(cfg, pm)
    if "capability" in a.stages:
        run_capability(cfg, pm)


if __name__ == "__main__":
    main()
