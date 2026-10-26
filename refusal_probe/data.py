import json
import os

from .config import DATA_DIR

HARMFUL_EVAL = ["advbench", "harmbench_test", "harmbench_val", "jailbreakbench", "strongreject"]


def load_split(harmtype: str, split: str, instructions_only=False):
    with open(os.path.join(DATA_DIR, "splits", f"{harmtype}_{split}.json")) as f:
        d = json.load(f)
    return [x["instruction"] for x in d] if instructions_only else d


def load_processed(name: str, instructions_only=False):
    with open(os.path.join(DATA_DIR, "processed", f"{name}.json")) as f:
        d = json.load(f)
    return [x["instruction"] for x in d] if instructions_only else d
