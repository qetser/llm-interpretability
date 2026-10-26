import os
from dataclasses import dataclass, field
from typing import Tuple

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(ROOT, "runs")


def list_models():
    with open(os.path.join(ROOT, "models.yaml")) as f:
        return [m["id"] for m in yaml.safe_load(f)["models"]]


def model_gb(model_id: str) -> float:
    with open(os.path.join(ROOT, "models.yaml")) as f:
        return next((m.get("gb", 8.0) for m in yaml.safe_load(f)["models"] if m["id"] == model_id), 8.0)


@dataclass
class Config:
    model_path: str
    n_train: int = 128
    n_val: int = 32
    n_test: int = 100
    filter_train: bool = True
    filter_val: bool = True
    max_new_tokens: int = 256
    gen_batch_size: int = 8
    harmful_eval_datasets: Tuple[str, ...] = ("jailbreakbench", "harmbench_test", "advbench", "strongreject")
    n_eval_per_dataset: int = 100
    kl_threshold: float = 0.1
    induce_refusal_threshold: float = 0.0
    prune_layer_percentage: float = 0.2
    lm_eval_tasks: Tuple[str, ...] = ("mmlu", "arc_challenge", "hellaswag", "truthfulqa_mc2", "winogrande", "gsm8k")
    lm_eval_batch_size: int = 8
    lm_eval_limit: int = None  # per-task example cap; None = full

    @property
    def model_alias(self) -> str:
        return os.path.basename(self.model_path)

    def artifact_path(self) -> str:
        return os.path.join(RUNS_DIR, self.model_alias)
