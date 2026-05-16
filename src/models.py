"""HuggingFace model loader and layer iterators."""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_model(model_path, dtype="bfloat16", device_map="balanced",
               max_memory=None, offload_folder=None):
    """Load a HuggingFace causal LM.

    device_map values:
      - "auto" / "balanced": dispatched by accelerate across visible GPUs.
      - None or "single": loaded on CPU, then moved to cuda:0. Use this
        when CUDA_VISIBLE_DEVICES already restricts to one GPU and you
        want the whole model on it.
    """
    from transformers import AutoModelForCausalLM
    logger.info(f"Loading model from {model_path}")
    torch_dtype = getattr(torch, dtype)
    if device_map == "auto" and torch.cuda.device_count() > 1:
        logger.info(f"Multi-GPU mode: {torch.cuda.device_count()} GPUs available")

    single_gpu = device_map in (None, "single")
    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=None if single_gpu else device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if offload_folder:
        kwargs["offload_folder"] = offload_folder
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if single_gpu and torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False
    model.eval()
    logger.info(f"Model loaded: {type(model).__name__}, dtype={torch_dtype}")
    return model


def find_linear_layers(model, exclude_patterns=None):
    """All Linear/quantized-Linear modules, excluding embeddings and the LM head."""
    if exclude_patterns is None:
        exclude_patterns = ["lm_head", "embed_tokens", "embed"]
    out = {}
    types = {"Linear", "CompressedLinear", "QuantizedLinear",
             "QLinear", "Int4Linear", "FP4Linear"}
    for name, m in model.named_modules():
        if type(m).__name__ not in types:
            continue
        if any(p in name for p in exclude_patterns):
            continue
        out[name] = m
    return out


def find_norm_layers(model):
    types = {"RMSNorm", "LayerNorm", "LlamaRMSNorm", "MistralRMSNorm", "Qwen2RMSNorm"}
    return {n: m for n, m in model.named_modules() if type(m).__name__ in types}


def get_tokenizer(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok
