"""Metrics shared between the search drivers and the evaluation paths.

Two groups of helpers live here:

  Training-loop losses (gradient enabled):
    compute_reference_log_probs  cache reference log P for KL.
    compute_kl_loss              KL(reference || model) over a batch.
    compute_ce_loss              standard next-token cross-entropy.
    compute_objective            kl / ce dispatcher.

  Post-hoc evaluation metrics (torch.no_grad):
    compute_perplexity           mean perplexity over a tokenized corpus.
    compute_baseline_topk        top-k logit values / indices for the
                                 baseline (full-precision) model.
    compute_all_metrics          NLL + top-k KL of a candidate model
                                 against the baseline topk cache.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from common import get_input_device

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Training-loop losses
# ----------------------------------------------------------------------------


@torch.no_grad()
def compute_reference_log_probs(model, calibration_data, batch_size=4):
    """Cache next-token reference log-probabilities for the calibration set.

    Returns:
        list of (B, T-1, V) torch.Tensor in float16 on CPU. Position t in
        each row holds log P(token_{t+1} | tokens_{<=t}) under the model.
    """
    logger.info("Computing reference log-probabilities...")
    device = get_input_device(model)
    ref_log_probs = []
    n = calibration_data.size(0)
    for i in range(0, n, batch_size):
        batch = calibration_data[i:i + batch_size].to(device)
        logits = model(input_ids=batch).logits[:, :-1, :].contiguous()
        ref_log_probs.append(F.log_softmax(logits.float(), dim=-1).half().cpu())
    total_tokens = sum(lp.numel() // lp.size(-1) for lp in ref_log_probs)
    mem_mb = sum(lp.nbytes for lp in ref_log_probs) / 1e6
    logger.info(f"Cached reference log-probs: {len(ref_log_probs)} batches, "
                f"{total_tokens} tokens, {mem_mb:.0f} MB")
    return ref_log_probs


def compute_kl_loss(model, input_ids, ref_log_probs, mask=None, topk=0):
    """KL(reference || model) summed over tokens, normalised over kept tokens.

    Args:
        model: forward returns logits.
        input_ids: (B, T) integer token ids.
        ref_log_probs: (B, T-1, V) reference log-probabilities for the batch.
        mask: (B, T) or None. If provided, marks loss-target token positions
            (e.g. the assistant turn in a chat dataset); only those
            contribute to the loss. If None, every position contributes.
        topk: if > 0, restrict the KL sum to the top-k reference tokens at
            each position (focuses gradient on the most probable tokens and
            avoids the long-tail noise). 0 means full vocabulary.

    Returns:
        Scalar Tensor with gradient enabled.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()
    ref_lp = ref_log_probs.to(logits.device)
    shift_mask = None
    if mask is not None:
        shift_mask = mask[:, 1:].contiguous().float().to(logits.device)

    B, T, V = logits.shape
    chunk = 128
    total_kl = 0.0
    total_tokens = 0.0

    for t in range(0, T, chunk):
        te = min(t + chunk, T)
        model_lp = F.log_softmax(logits[:, t:te, :].float(), dim=-1)
        rp = ref_lp[:, t:te, :]

        if topk > 0 and topk < V:
            _, tk_idx = rp.topk(topk, dim=-1)
            rp_k = rp.gather(-1, tk_idx)
            mp_k = model_lp.gather(-1, tk_idx)
            rp_k_norm = rp_k - rp_k.logsumexp(dim=-1, keepdim=True)
            kl = (rp_k_norm.exp() * (rp_k_norm - mp_k)).sum(dim=-1)
        else:
            kl = (rp.exp() * (rp - model_lp)).sum(dim=-1)

        if shift_mask is not None:
            m_chunk = shift_mask[:, t:te]
            total_kl = total_kl + (kl * m_chunk).sum()
            total_tokens = total_tokens + m_chunk.sum()
        else:
            total_kl = total_kl + kl.sum()
            total_tokens = total_tokens + kl.numel()

    if isinstance(total_tokens, torch.Tensor):
        total_tokens = total_tokens.clamp(min=1)
    elif total_tokens < 1:
        total_tokens = 1
    return total_kl / total_tokens


def compute_ce_loss(model, input_ids, mask=None):
    """Next-token cross-entropy. mask, if given, selects loss-target positions."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous().to(logits.device)
    V = logits.size(-1)
    if mask is not None:
        shift_mask = mask[:, 1:].contiguous().float().to(logits.device)
        per_tok = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                                  reduction='none').view(labels.shape)
        return (per_tok * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return F.cross_entropy(logits.view(-1, V), labels.view(-1))


def compute_objective(model, input_ids, objective, ref_log_probs=None, mask=None,
                      topk=0):
    """Dispatch to compute_kl_loss or compute_ce_loss by objective name."""
    if objective == 'kl':
        return compute_kl_loss(model, input_ids, ref_log_probs, mask=mask,
                               topk=topk)
    if objective == 'ce':
        return compute_ce_loss(model, input_ids, mask=mask)
    raise ValueError(f"Unknown objective {objective!r}; expected 'kl' or 'ce'.")


# ----------------------------------------------------------------------------
# Post-hoc evaluation metrics
# ----------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(model, data, batch_size=1, masks=None):
    """Mean perplexity, variance, and per-sample PPL across data."""
    n = len(data)
    device = next(model.parameters()).device
    use_masks = masks is not None
    nlls: List[float] = []

    for i in trange(0, n, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, n)
        x = torch.cat(data[i:j]).to(device)
        logits = model(x).logits[:, :-1, :].contiguous()
        labels = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        if use_masks:
            m = torch.cat(masks[i:j]).to(device)[:, 1:].float()
            valid = m.sum(dim=1).clamp(min=1)
            per_seq = (loss * m).sum(dim=1) / valid
        else:
            per_seq = loss.mean(dim=1)
        nlls.extend(per_seq.cpu().tolist())

    mean_nll = np.mean(nlls)
    per_ppl = [np.exp(v) for v in nlls]
    return float(np.exp(mean_nll)), float(np.var(per_ppl)), per_ppl


@torch.no_grad()
def compute_baseline_topk(model, input_ids, batch_size=4, top_k=10, masks=None):
    """Top-k baseline logit values and indices, used as KL reference."""
    model.eval()
    device = next(model.parameters()).device
    vals, idx = [], []
    for i in range(0, input_ids.shape[0], batch_size):
        b = input_ids[i:i + batch_size].to(device)
        logits = model(b).logits[:, :-1, :].float()
        v, k = logits.topk(top_k, dim=-1)
        vals.append(v.cpu()); idx.append(k.cpu())
        del logits; torch.cuda.empty_cache()
    return torch.cat(vals, dim=0), torch.cat(idx, dim=0)


@torch.no_grad()
def compute_all_metrics(model, calibration_data, baseline_topk_vals,
                        baseline_topk_idx, batch_size=4, top_k=10,
                        temperature=1.0, masks=None):
    """Mean NLL and KL-to-baseline (top-k) over the calibration set."""
    model.eval()
    device = next(model.parameters()).device
    use_masks = masks is not None

    total_nll = 0.0
    total_kl = 0.0
    total_tokens = 0
    bidx = 0

    for i in range(0, calibration_data.shape[0], batch_size):
        b = calibration_data[i:i + batch_size].to(device)
        bs = b.shape[0]
        if use_masks:
            shift_masks = masks[i:i + batch_size].to(device)[:, 1:].float()

        logits = model(b).logits
        sl = logits[:, :-1, :].contiguous()
        labels = b[:, 1:].contiguous()

        nll = F.cross_entropy(
            sl.view(-1, sl.size(-1)), labels.view(-1),
            reduction="none",
        ).view(bs, -1)
        if use_masks:
            total_nll += (nll * shift_masks).sum().item()
        else:
            total_nll += nll.sum().item()

        dl = sl.float() / temperature
        vtl = baseline_topk_vals[bidx:bidx + bs].to(device) / temperature
        vti = baseline_topk_idx[bidx:bidx + bs].to(device)

        for s in range(0, dl.shape[1], 256):
            e = min(s + 256, dl.shape[1])
            dlc = dl[:, s:e, :]
            vtlc = vtl[:, s:e, :]
            vtic = vti[:, s:e, :]
            dtl = dlc.gather(dim=-1, index=vtic)
            vtp = F.softmax(vtlc, dim=-1)
            dtp = F.softmax(dtl, dim=-1)
            kl = (vtp * (torch.log(vtp + 1e-10) - torch.log(dtp + 1e-10))).sum(dim=-1)
            if use_masks:
                cm = shift_masks[:, s:e]
                total_kl += (kl * cm).sum().item()
                total_tokens += cm.sum().item()
            else:
                total_kl += kl.sum().item()
                total_tokens += kl.numel()

        bidx += bs
        torch.cuda.empty_cache()

    n = total_tokens if total_tokens > 0 else 1
    return total_nll / n, total_kl / n
