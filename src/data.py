"""Calibration data loaders.

Plain datasets (return List[Tensor], one (1, seq_length) chunk per entry):
  wikitext2, c4, fineweb_edu.

Chat datasets (return (seqs, masks) stacked tensors; mask=1.0 on assistant
answer tokens, 0.0 on prompt and padding):
  evol_codealpaca (code), tulu_math (math).
"""

import os
import random
import logging
from typing import List

import torch
from datasets import load_dataset
from tqdm import trange

logger = logging.getLogger(__name__)


# Chat-style datasets ship answer masks alongside token sequences. Search
# drivers that care about answer-only loss check this set to unpack the
# return value as (seqs, masks).
MASKED_DATASETS: set = {"evol_codealpaca", "tulu_math"}


def get_wikitext2(num_samples, seq_length, tokenizer, train=True, seed=42):
    logger.info("Loading WikiText-2")
    random.seed(seed)
    if train:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokens = tokenizer("\n\n".join(ds["text"]),
                           return_tensors="pt",
                           add_special_tokens=False).input_ids
        out = []
        for _ in trange(num_samples, desc="Preparing data"):
            i = random.randint(0, tokens.shape[1] - seq_length - 1)
            out.append(tokens[:, i:i + seq_length])
        return out
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokens = tokenizer("\n\n".join(ds["text"]),
                       return_tensors="pt",
                       add_special_tokens=False).input_ids
    n = tokens.numel() // seq_length
    return [tokens[:, i * seq_length:(i + 1) * seq_length] for i in range(n)]


def get_fineweb_edu(num_tokens, seq_length, tokenizer, train=True, seed=42):
    logger.info("Loading FineWeb-Edu")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    out = []
    num_samples = num_tokens // seq_length
    buf = torch.tensor([], dtype=torch.int64).reshape(1, 0)
    for sample in ds:
        if not sample["text"].strip():
            continue
        tok = tokenizer(sample["text"], return_tensors="pt",
                        add_special_tokens=False).input_ids
        buf = torch.cat([buf, tok], dim=1)
        while buf.shape[1] >= seq_length:
            out.append(buf[:, :seq_length])
            buf = buf[:, seq_length:]
            if len(out) >= num_samples:
                break
        if len(out) >= num_samples:
            break
    logger.info(f"Loaded {len(out)} sequences of length {seq_length}")
    return out


def get_c4(num_samples, seq_length, tokenizer, train=True, seed=42):
    logger.info("Loading C4")
    if train:
        ds = load_dataset(
            "allenai/c4", "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87")
    else:
        ds = load_dataset(
            "allenai/c4", "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation[:1100]",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87")
    random.seed(seed)
    out = []
    cur = torch.tensor([], dtype=torch.int64)
    pbar = trange(num_samples, desc="Preparing C4 data")
    n = 0
    for sample in iter(ds):
        tok = tokenizer(sample["text"], return_tensors="pt",
                        add_special_tokens=False).input_ids
        cur = torch.cat([cur, tok], dim=1)
        if cur.numel() >= seq_length:
            n += 1
            pbar.update()
            out.append(cur[:, :seq_length])
            cur = torch.tensor([], dtype=torch.int64)
        else:
            nl = tokenizer("\n\n", return_tensors="pt",
                           add_special_tokens=False).input_ids
            cur = torch.cat([cur, nl], dim=1)
        if n >= num_samples:
            break
    pbar.close()
    return out


def _load_chat_dataset(name, n_samples, seq_length, tokenizer, seed=42):
    """Load a chat-formatted calibration dataset and build answer-only masks.

    Returns (seqs, masks): (N, seq_length) int tensor and a matching float
    tensor where mask=1.0 marks positions inside the assistant turn (loss
    target) and 0.0 marks the prompt / padding.
    """
    random.seed(seed)
    if name == "evol_codealpaca":
        ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        raw_samples = []
        for i in indices[:n_samples * 4]:
            raw_samples.append({
                "question": [{"role": "user", "content": ds[i]["instruction"]}],
                "full": [
                    {"role": "user", "content": ds[i]["instruction"]},
                    {"role": "assistant", "content": ds[i]["output"]},
                ],
            })
    elif name == "tulu_math":
        ds = load_dataset("allenai/tulu-3-sft-personas-math", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        raw_samples = []
        for i in indices[:n_samples * 4]:
            msgs = ds[i]["messages"]
            # Question = everything up to the last assistant turn.
            q_msgs = []
            for m in msgs:
                if m["role"] == "assistant":
                    break
                q_msgs.append(m)
            raw_samples.append({"question": q_msgs, "full": msgs})
    else:
        raise ValueError(f"Unknown chat dataset: {name}")

    has_chat = hasattr(tokenizer, "apply_chat_template")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    all_seqs = []
    all_masks = []
    for sample in raw_samples:
        if len(all_seqs) >= n_samples:
            break
        if has_chat:
            q_result = tokenizer.apply_chat_template(
                sample["question"], return_tensors="pt",
                add_generation_prompt=True)
            full_result = tokenizer.apply_chat_template(
                sample["full"], return_tensors="pt",
                add_generation_prompt=False)
            # apply_chat_template may return a tensor or a BatchEncoding.
            if hasattr(q_result, "shape"):
                q_ids = q_result[0] if q_result.dim() == 2 else q_result
            else:
                q_ids = q_result["input_ids"][0]
            if hasattr(full_result, "shape"):
                full_ids = full_result[0] if full_result.dim() == 2 else full_result
            else:
                full_ids = full_result["input_ids"][0]
        else:
            q_text = "\n".join(m["content"] for m in sample["question"])
            q_ids = tokenizer(q_text, add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0]
            full_text = "\n".join(m["content"] for m in sample["full"])
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]

        q_len = q_ids.shape[0]
        f_len = full_ids.shape[0]

        if f_len < 10:  # skip degenerate samples
            continue

        mask = torch.zeros(seq_length)
        if f_len >= seq_length:
            seq = full_ids[:seq_length]
            mask[q_len:seq_length] = 1.0
        else:
            seq = torch.full((seq_length,), pad_id, dtype=full_ids.dtype)
            seq[:f_len] = full_ids
            mask[q_len:f_len] = 1.0

        if mask.sum() < 5:  # skip if answer portion is too short
            continue

        all_seqs.append(seq)
        all_masks.append(mask)

    seqs = torch.stack(all_seqs[:n_samples])
    masks = torch.stack(all_masks[:n_samples])
    avg_ans = masks.sum(1).mean().item()
    logger.info(f"Loaded {seqs.shape[0]} sequences of length {seq_length} "
                f"from {name} (avg answer tokens: {avg_ans:.0f})")
    return seqs, masks


def get_data(name, num_tokens, seq_length, tokenizer, train=True, seed=1):
    """Load calibration tokens by dataset name or from a saved .pt file.

    Plain text datasets return List[Tensor] (one (1, seq_length) chunk each).
    Chat datasets (in MASKED_DATASETS) return (seqs, masks) stacked tensors.
    """
    if os.path.isfile(name):
        d = torch.load(name)[: num_tokens // seq_length]
        return [s[:, :seq_length] for s in d]
    n_samples = num_tokens // seq_length
    if name == "fineweb_edu":
        return get_fineweb_edu(num_tokens, seq_length, tokenizer, train, seed)
    if name == "c4":
        return get_c4(n_samples, seq_length, tokenizer, train, seed)
    if name == "wikitext2":
        return get_wikitext2(n_samples, seq_length, tokenizer, train, seed)
    if name in MASKED_DATASETS:
        return _load_chat_dataset(name, n_samples, seq_length, tokenizer, seed)
    raise ValueError(f"Unknown dataset: {name}")


def load_calibration_data(name, n_samples, seq_length, tokenizer, seed=42):
    """Load calibration sequences and a per-token answer mask.

    Plain text datasets (wikitext2, c4, fineweb_edu) get an all-ones mask.
    Chat datasets in MASKED_DATASETS (evol_codealpaca, tulu_math) get the
    answer-only mask produced by their loader. Names joined with '+' mix
    multiple sources at equal sample count (e.g. "fineweb_edu+evol_codealpaca")
    and are shuffled once to interleave the source order.

    Returns:
        seqs: (N, seq_length) token IDs.
        masks: (N, seq_length) float tensor, 1.0 marks loss-target tokens.
    """
    if "+" in name:
        parts = name.split("+")
        n_per = n_samples // len(parts)
        all_seqs, all_masks = [], []
        for i, part in enumerate(parts):
            s, m = load_calibration_data(
                part.strip(), n_per, seq_length, tokenizer, seed + i * 1000)
            all_seqs.append(s)
            all_masks.append(m)
        seqs = torch.cat(all_seqs, dim=0)
        masks = torch.cat(all_masks, dim=0)
        perm = torch.randperm(seqs.shape[0],
                              generator=torch.Generator().manual_seed(seed))
        seqs = seqs[perm]
        masks = masks[perm]
        logger.info(f"Mixed dataset '{name}': {seqs.shape[0]} total seqs")
        return seqs, masks

    cal = get_data(name, n_samples * seq_length, seq_length,
                   tokenizer, train=True, seed=seed)
    if name in MASKED_DATASETS:
        seqs, masks = cal
        return seqs, masks
    seqs = torch.cat(cal, dim=0)
    masks = torch.ones(seqs.shape[0], seqs.shape[1])
    return seqs, masks
