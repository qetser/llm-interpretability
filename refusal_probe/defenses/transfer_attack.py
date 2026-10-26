"""Attack a defended model with the refusal direction extracted from the undefended model (no re-estimation)."""
import argparse
import json
import os

import torch

from ..bench import safety
from ..config import RUNS_DIR, Config
from ..model import ProbedModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="defended model directory")
    ap.add_argument("--base-alias", default="Qwen2.5-0.5B-Instruct", help="runs/<alias> holding the undefended direction")
    ap.add_argument("--n-eval", type=int, default=100)
    a = ap.parse_args()

    cfg = Config(model_path=a.model, n_eval_per_dataset=a.n_eval)
    meta = json.load(open(f"{RUNS_DIR}/{a.base_alias}/direction_metadata.json"))
    direction = torch.load(f"{RUNS_DIR}/{a.base_alias}/direction.pt")
    pm = ProbedModel(a.model)
    datasets = {n: safety.sample(n, cfg.n_eval_per_dataset) for n in cfg.harmful_eval_datasets}
    out = os.path.join(cfg.artifact_path(), "safety_transfer")
    res = safety.eval_conditions(pm, {"ablation_base_direction": pm.ablation_hooks(direction)}, datasets, out, cfg.max_new_tokens, cfg.gen_batch_size)
    json.dump(res, open(f"{out}/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
