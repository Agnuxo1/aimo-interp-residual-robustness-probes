"""Residual-stream feature extraction shared by training and the submission.

The same functions run on the training server and inside the Codabench
bundle, so the probe always sees identically computed activations.
"""
from __future__ import annotations

import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = "Please reason step by step, and put your final answer within \boxed{}."


def safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, problem: str) -> str:
    problem = problem.replace("\r\n", "\n").replace("\r", "\n").strip()
    messages = [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"]


@torch.inference_mode()
def encode(tokenizer, model, problem: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (last-token, mean-over-tokens) residual streams, shape [layers+1, d]."""
    prompt = build_prompt(tokenizer, problem)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**ids, output_hidden_states=True, use_cache=False)
    hs = torch.stack([h[0] for h in out.hidden_states]).float()  # [L+1, T, d]
    return hs[:, -1, :].cpu().numpy(), hs.mean(dim=1).cpu().numpy()
