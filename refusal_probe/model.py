import functools
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .hooks import add_hooks, build_ablation_hooks, build_addition_hooks


class ProbedModel:
    """Generic chat-model wrapper built on the tokenizer's chat template."""

    def __init__(self, model_path: str, dtype=torch.bfloat16, device_map="auto"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        ).eval()
        self.model.requires_grad_(False)

        inner = self.model.model
        self.blocks = inner.layers
        self.attns = [self._find(b, ("self_attn", "attn", "attention")) for b in self.blocks]
        self.mlps = [self._find(b, ("mlp", "feed_forward")) for b in self.blocks]
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks: List[int] = []  # filled by refusal_tokens.detect_refusal_tokens

    @staticmethod
    def _find(block, names):
        for n in names:
            if hasattr(block, n):
                return getattr(block, n)
        raise AttributeError(f"none of {names} on {type(block).__name__}")

    @property
    def device(self):
        return self.model.device

    # ---- prompt formatting -------------------------------------------------
    def format_prompt(self, instruction: str, history=None, system: str = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += list(history or [])
        msgs.append({"role": "user", "content": instruction})
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def tokenize(self, instructions: List[str], **kw):
        prompts = [self.format_prompt(i, **kw) for i in instructions]
        # templates already contain BOS where needed
        return self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)

    def _get_eoi_toks(self):
        """Tokens after the user instruction (template suffix): the positions we probe."""
        marker = "XXXMARKERXXX"
        full = self.format_prompt(marker)
        suffix = full.split(marker)[-1]
        return self.tokenizer.encode(suffix, add_special_tokens=False)

    # ---- interventions -----------------------------------------------------
    def ablation_hooks(self, direction):
        return build_ablation_hooks(self.blocks, self.attns, self.mlps, direction)

    def addition_hooks(self, layer, vector, coeff):
        return build_addition_hooks(self.blocks, layer, vector, coeff)

    # ---- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(self, instructions: List[str], pre_hooks=(), hooks=(), batch_size=8, max_new_tokens=256, progress=True):
        from tqdm import tqdm

        cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        cfg.pad_token_id = self.tokenizer.pad_token_id
        out = []
        rng = range(0, len(instructions), batch_size)
        for i in (tqdm(rng, desc="generate") if progress else rng):
            batch = instructions[i : i + batch_size]
            toks = self.tokenize(batch)
            with add_hooks(list(pre_hooks), list(hooks)):
                gen = self.model.generate(
                    input_ids=toks.input_ids.to(self.device),
                    attention_mask=toks.attention_mask.to(self.device),
                    generation_config=cfg,
                )
            gen = gen[:, toks.input_ids.shape[-1] :]
            out += [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
        return out

    def unload(self):
        del self.model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
