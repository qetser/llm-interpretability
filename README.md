# llm-interpretability

Refusal-direction research on open instruct models: how a single direction in the residual stream mediates refusal
([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)), how easily that direction can be removed, and techniques open
models can use to resist its removal. Includes a benchmark suite and a React chat UI for the probed models.

## Findings: refusal is mediated by one direction

For each model we extract a difference-in-means refusal direction, then ablate it (all layers, all token positions) and
measure refusal (substring match) on JailbreakBench, HarmBench, AdvBench and StrongREJECT (100 prompts each, averaged).

| Model | Direction (layer) | Refusal, no attack | Refusal, direction ablated | Over-refusal (harmless) | MMLU base -> ablated |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 17 | 0.73 | 0.00 | 0.04 | 0.477 -> 0.476 |
| Qwen2.5-1.5B-Instruct | 21 | 0.96 | 0.02 | 0.05 | 0.626 -> 0.624 |
| OLMo-2-0425-1B-Instruct | 8 | 0.86 | 0.04 | 0.04 | 0.419 -> 0.419 |

Removing one direction takes refusal from 73-96% to 0-4% while general capability (MMLU, ARC, HellaSwag, TruthfulQA,
Winogrande, GSM8K) is essentially unchanged. Adding the direction to harmless prompts induces refusal (94% on
Qwen2.5-0.5B). Full per-dataset, per-category numbers are in `runs/<model>/safety/summary.json`, capability scores in
`runs/<model>/capability/lm_eval.json`, and direction-selection plots in `runs/<model>/select_direction/`.

## Hardening: making refusal resistant to direction removal

`refusal_probe/defenses/` implements techniques from recent work on keeping open-weight models from losing refusal to
direction ablation ("abliteration"):

1. **Diverse refusal openers** (`diverse_refusal.py`, after [Labunets 2026](https://arxiv.org/abs/2608.25390)).
   Refusal training that always starts the same way concentrates the refusal signal in a low-rank direction.
   Fine-tuning on refusals with `m` distinct first tokens (plus the model's own harmless completions as retention data)
   spreads it over more directions. `--m 1` is the matched control.
2. **Decoy Direction Optimization** (`ddo.py`, after [Muhamed et al. 2026](https://arxiv.org/abs/2609.16204)).
   A few low-norm MLP neurons per layer read the refusal coordinate and write a learned decoy orthogonal to the true
   refusal direction, so difference-in-means estimates the decoy. Trained with refusal, retention and a differentiable
   self-ablation confusion loss, then folded back into plain weights (no runtime overhead).
3. **Attacker-aware evaluation** (`transfer_attack.py`, `summarize.py`). Each defended model is attacked two ways:
   re-estimating the direction on the defended model with the full pipeline (adaptive attacker), and ablating the
   direction taken from the undefended model (stale-direction attacker). Refusal, over-refusal and capability are
   reported side by side.

Further reading used for these: [Fool's Gold](https://www.alphaxiv.org/abs/2608.17202),
[Bait-and-Recover](https://www.alphaxiv.org/abs/2609.05794), [Refusal Aliases](https://www.alphaxiv.org/abs/2608.18093),
[Tamper-Resistant Safeguards](https://arxiv.org/abs/2408.00761), [Deep Ignorance](https://arxiv.org/abs/2508.06601).

```bash
python -m refusal_probe.defenses.diverse_refusal --m 16 --out defended/qwen0.5b-div16
python -m refusal_probe.defenses.diverse_refusal --m 1  --out defended/qwen0.5b-div1     # control
python -m refusal_probe.defenses.ddo --out defended/qwen0.5b-ddo
python -m refusal_probe.pipeline --model defended/qwen0.5b-div16 --stages extract safety capability   # adaptive attacker
python -m refusal_probe.defenses.transfer_attack --model defended/qwen0.5b-div16                      # stale-direction attacker
python -m refusal_probe.defenses.summarize Qwen2.5-0.5B-Instruct qwen0.5b-div1 qwen0.5b-div16 qwen0.5b-ddo
```

## Implementation notes

Differences from the reference implementation (`../refusal_direction`):
- **One generic model class** built on each tokenizer's chat template, instead of a class per family. Adding a model is one line in `models.yaml`.
- **Refusal tokens are detected automatically** (first tokens that reliably start refusals) instead of hand-picked.
- **Interventions are hooks**, so the same loaded model serves baseline / ablation / steering and lm-eval can run under them.
- Direction selection **relaxes the KL / steering thresholds stepwise** if nothing passes (small models often fail the paper's `KL<0.1`); what was relaxed is recorded in `direction_metadata.json`.

## Setup
```bash
./setup.sh      # .venv (py3.11, torch 2.6 cu124), local Node in .node, builds web/
```
The GPU here is MIG-partitioned; `scripts/env.sh` pins one slice. Pick one with free memory:
`MIG_DEVICE=<uuid from nvidia-smi -L> scripts/run_all.sh`.

## Probe + benchmark
```bash
scripts/run_all.sh                                   # every model in models.yaml
scripts/run_all.sh Qwen/Qwen2.5-1.5B-Instruct        # one model
scripts/run_all.sh -- --lm-eval-limit 200 --judge    # extra pipeline flags
python -m refusal_probe.pipeline --model X --stages safety loss   # re-run selected stages
```
Stages (artifacts in `runs/<model>/`):
1. `extract` - refusal tokens, candidate directions (every layer x post-instruction position), selection -> `direction.pt`, `direction_metadata.json`, `select_direction/*.png`
2. `safety` - refusal rate on JailbreakBench, HarmBench, AdvBench, StrongREJECT under **baseline / ablation / -steer**; over-refusal on harmless Alpaca prompts and refusal induced by **+steer**. Substring matching always; `--judge` adds the HarmBench Mistral-7B classifier for harmful compliance (needs ~14 GB extra).
3. `loss` - cross-entropy on the model's own harmless completions (baseline vs ablation vs steer)
4. `capability` - lm-evaluation-harness: MMLU, ARC-Challenge, HellaSwag, TruthfulQA-MC2, Winogrande, GSM8K, baseline vs ablated (hooks are active during evaluation).

Per-prompt completions are not committed (`.gitignore`); re-run the stages to regenerate them.

## Chat UI
```bash
scripts/serve.sh                # http://127.0.0.1:8000 (UI + gateway); HOST=0.0.0.0 to expose
GPUS="MIG-... MIG-..." scripts/serve.sh   # choose slices (default: all MIG slices / GPUs from nvidia-smi -L)
cd web && npm run dev           # UI dev server on :5173, proxies /api
```
`serve.sh` starts one worker per GPU slice (`:8001+`, each keeps one model resident) behind a gateway on `:8000` that
serves the UI and load-balances chat: warm idle worker first, else an idle worker with enough free memory
(`gb` in `models.yaml`), else queue on the least-loaded. Free memory is read through NVML, so idle workers hold no GPU memory.
If no slice has room, the UI shows an error instead of crashing. Stopping a response in the UI cancels generation on the GPU.
`/api/workers` and the sidebar show per-GPU free memory and load.
Pick a probed model, then Baseline / Ablate refusal / Steer (strength slider), optionally side-by-side with baseline.
