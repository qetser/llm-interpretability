"""Table of refusal rate (substring) per model: no attack vs re-estimated-direction ablation vs undefended-direction ablation."""
import json
import os
import sys

import numpy as np

from ..config import RUNS_DIR

DATASETS = ["jailbreakbench", "harmbench_test", "advbench", "strongreject"]


def _mean(path, cond):
    if not os.path.exists(path):
        return None
    s = json.load(open(path))
    return float(np.mean([s[d][cond]["refusal_rate_substring"] for d in DATASETS if d in s and cond in s[d]]))


def main(aliases):
    f = lambda x: "  -  " if x is None else f"{x:5.2f}"
    print(f"{'model':32} {'no-attack':>9} {'ablate(re-est)':>14} {'ablate(base dir)':>16} {'over-refusal':>12} {'arc':>5} {'hellaswag':>9}")
    for al in aliases:
        d = f"{RUNS_DIR}/{al}"
        h = json.load(open(f"{d}/safety/summary.json"))["harmless"]["baseline"]["refusal_rate_substring"] if os.path.exists(f"{d}/safety/summary.json") else None
        cap = {}
        if os.path.exists(f"{d}/capability/lm_eval.json"):
            cap = json.load(open(f"{d}/capability/lm_eval.json")).get("baseline", {}).get("headline", {})
        g = lambda k: next((v for kk, v in cap.items() if k in kk), None)
        print(f"{al:32} {f(_mean(f'{d}/safety/summary.json', 'baseline')):>9} {f(_mean(f'{d}/safety/summary.json', 'ablation')):>14} "
              f"{f(_mean(f'{d}/safety_transfer/summary.json', 'ablation_base_direction')):>16} {f(h):>12} {f(g('arc')):>5} {f(g('hellaswag')):>9}")


if __name__ == "__main__":
    main(sys.argv[1:])
