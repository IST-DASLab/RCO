"""RCO MoE expert-pruning search.

Each expert is a binary group (keep/prune) with costs [0, 1]. The budget
fixes the total number of pruned experts. STE-masked forward through the
MoE block, Gumbel sampling, gradient projection onto the budget tangent
plane, Adam step, scalar-shift retraction.
"""

import logging
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common import get_input_device
from data import load_calibration_data
from manifold import (
    project_gradient,
    retraction,
    vector_transport,
    project_gradient_per_layer,
    retraction_per_layer,
    vector_transport_per_layer,
)
from metrics import (
    compute_kl_loss,
    compute_reference_log_probs as _compute_reference_log_probs,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def get_moe_info(model):
    """Extract MoE architecture info."""
    c = model.config
    return c.num_hidden_layers, c.num_experts, c.num_experts_per_tok


def build_ref_cache(model, data, masks, batch_size=4):
    """Cache reference log-probs and the calibration mask sliced per batch.

    Returns two parallel lists of length ceil(N / batch_size):
        ref_log_probs: list of (B, T-1, V) float16 Tensors on CPU.
        ref_masks:     list of (B, T) float Tensors on CPU. Unshifted, since
                       common.compute_kl_loss applies the next-token shift.
    """
    ref_lps = _compute_reference_log_probs(model, data, batch_size)
    ref_masks = []
    n = data.size(0)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        ref_masks.append(masks[i:end].float().cpu())
    return ref_lps, ref_masks


# ----------------------------------------------------------------------------
# STE masks
# ----------------------------------------------------------------------------


def budget_assignment(logits, budget):
    """
    For N binary groups with costs [0, 1], select exactly budget to prune.

    Args:
        logits: (N, 2) tensor. [:, 0] = keep logit, [:, 1] = prune logit.
        budget: int, number to prune.

    Returns: (N,) long tensor, 0=keep 1=prune.
    """
    advantage = logits[:, 1] - logits[:, 0]
    _, top_idx = advantage.topk(budget)
    assignment = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)
    assignment[top_idx] = 1
    return assignment


def compute_ste_masks(alpha, tau, budget):
    """
    Gumbel noise + budget-feasible hard assignment + STE.

    Args:
        alpha: (N, 2) logits, requires_grad=True
        tau: Gumbel-softmax temperature
        budget: int, number of experts to prune

    Returns:
        ste_keep: (N,) survival mask with STE gradient.
            Forward value is hard 0/1. Backward flows through soft probs.
        hard_assignment: (N,) long, 0=keep 1=prune
    """
    u = torch.rand_like(alpha).clamp(1e-20)
    gumbel = -torch.log(-torch.log(u) + 1e-20)
    noisy = (alpha + gumbel) / tau

    soft_probs = F.softmax(noisy, dim=1)
    soft_keep = soft_probs[:, 0]

    hard_asgn = budget_assignment(noisy.detach(), budget)
    hard_keep = 1.0 - hard_asgn.float()

    # straight-through: forward uses hard, backward uses soft
    ste_keep = hard_keep + (soft_keep - soft_keep.detach())

    return ste_keep, hard_asgn


# ----------------------------------------------------------------------------
# MoE forward patching
# ----------------------------------------------------------------------------


class MoEPruneWrapper:
    """Patches a SparseMoeBlock to scale expert outputs by STE masks.

    The STE mask is a per-expert scalar: 1.0 (keep) or 0.0 (prune) in the
    forward pass, with gradients flowing through the soft Gumbel-softmax
    probabilities in the backward pass.

    Expert outputs are multiplied by the mask AFTER the expert computation
    and routing weight application. This is functionally equivalent to
    removing the expert from the router (evaluation uses proper -inf masking).

    Supports models with shared experts (e.g. Qwen3-Next) by passing
    their output through unchanged.
    """

    def __init__(self, moe_block, layer_idx, n_experts, shared_state):
        self.block = moe_block
        self.layer_idx = layer_idx
        self.n_experts = n_experts
        self.shared = shared_state
        self._orig_forward = moe_block.forward
        self.has_shared_expert = hasattr(moe_block, 'shared_expert')
        try:
            import inspect
            src = inspect.getsource(moe_block.__class__.forward)
            self._returns_router_logits = 'router_logits' in src
        except (OSError, TypeError):
            self._returns_router_logits = False

        wrapper_ref = self

        def patched_forward(hidden_states):
            return wrapper_ref._forward(hidden_states)

        self.block.forward = patched_forward

    def _forward(self, hidden_states):
        block = self.block
        ste_masks = self.shared['ste_masks']
        layer_mask = ste_masks[self.layer_idx]

        B, S, D = hidden_states.shape
        h = hidden_states.view(-1, D)

        router_output = block.gate(h)
        if isinstance(router_output, (tuple, list)):
            routing_weights_full = router_output[0]
            routing_weights = router_output[1]
            selected = router_output[2]
        else:
            # stock HF MoE gates return raw logits and expose top_k /
            # norm_topk_prob on the block, not the gate module.
            top_k = getattr(block.gate, 'top_k', None)
            if top_k is None:
                top_k = getattr(block, 'top_k', None)
            if top_k is None:
                top_k = getattr(block, 'num_experts_per_tok', None)
            assert top_k is not None, \
                "Could not determine top_k from gate or MoE block."
            norm_topk_prob = getattr(block.gate, 'norm_topk_prob',
                                     getattr(block, 'norm_topk_prob', False))
            routing_weights_full = F.softmax(router_output, dim=1, dtype=torch.float)
            routing_weights, selected = torch.topk(
                routing_weights_full, top_k, dim=-1)
            if norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # scale each selected expert's routing weight by its survival prob
        # (1.0 keep, 0.0 prune), then renormalize so the row still sums to 1.
        lm = layer_mask
        if lm.device != h.device:
            lm = lm.to(h.device)

        expert_survival = lm[selected]
        ste_weights = routing_weights.float() * expert_survival.float()
        ste_weights = ste_weights / ste_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        ste_weights = ste_weights.to(h.dtype)

        # stash raw gate logits for the OLMoE-style return signature.
        gate_logits_for_return = router_output if not isinstance(
            router_output, (tuple, list)) else None

        if isinstance(block.experts, nn.ModuleList):
            # stock HF layout (Qwen3-Next, Mixtral-style): iterate hit experts.
            n_experts = len(block.experts)
            expert_output = torch.zeros_like(h)
            expert_mask_oh = F.one_hot(
                selected, num_classes=n_experts
            ).permute(2, 1, 0)
            hit = expert_mask_oh.sum(dim=(-1, -2)).nonzero(as_tuple=True)[0]
            for e_idx in hit.tolist():
                idx, top_x = torch.where(expert_mask_oh[e_idx])
                if top_x.numel() == 0:
                    continue
                current = h[top_x]
                out = block.experts[e_idx](current)
                out = out * ste_weights[top_x, idx, None]
                expert_output.index_add_(0, top_x, out.to(expert_output.dtype))
        else:
            # fused-expert layout takes (hidden, indices, weights) directly.
            expert_output = block.experts(h, selected, ste_weights)

        if self.has_shared_expert:
            shared_out = block.shared_expert(h)
            shared_out = torch.sigmoid(block.shared_expert_gate(h)) * shared_out
            expert_output = expert_output + shared_out

        result = expert_output.view(B, S, D)
        if self._returns_router_logits and gate_logits_for_return is not None:
            return result, gate_logits_for_return
        return result

    def restore(self):
        self.block.forward = self._orig_forward


def install_wrappers(model, n_layers, n_experts, shared_state):
    wrappers = []
    for l in range(n_layers):
        w = MoEPruneWrapper(model.model.layers[l].mlp,
                            l, n_experts, shared_state)
        wrappers.append(w)
    return wrappers


def remove_wrappers(wrappers):
    for w in wrappers:
        w.restore()


# ----------------------------------------------------------------------------
# Manifold ops (alpha-space)
# ----------------------------------------------------------------------------


def get_expected_budget(alpha, costs):
    with torch.no_grad():
        p = torch.softmax(alpha, dim=1)
        return (p * costs.unsqueeze(0)).sum().item()


def init_alpha(alpha, target, N):
    """Initialize alpha so E[budget] = target.

    With costs [0, 1] and alpha[:] = [0, c]:
        softmax([0, c])[1] = sigmoid(c) = target/N
        c = log(target / (N - target))
    """
    frac = target / N
    c = math.log(frac / (1.0 - frac + 1e-10))
    with torch.no_grad():
        alpha[:, 0] = 0.0
        alpha[:, 1] = c
    costs = torch.tensor([0.0, 1.0], device=alpha.device)
    actual = get_expected_budget(alpha, costs)
    logger.info(f"Alpha init: c={c:.4f}, E[budget]={actual:.2f} (target={target})")


def budget_assignment_per_layer(logits, target_per_layer, n_layers, n_experts):
    """Per-layer top-k for discrete feasibility (uniform k per layer)."""
    advantage = logits[:, 1] - logits[:, 0]
    adv_ll = advantage.view(n_layers, n_experts)
    _, top_idx = adv_ll.topk(target_per_layer, dim=1)
    assignment = torch.zeros(n_layers, n_experts, dtype=torch.long,
                             device=logits.device)
    assignment.scatter_(1, top_idx, 1)
    return assignment.view(-1)


# ----------------------------------------------------------------------------
# Hard pruning evaluation
# ----------------------------------------------------------------------------


def apply_hard_pruning_mask(model, prune_mask):
    """
    Apply hard expert pruning by masking router logits to -inf.

    Args:
        prune_mask: (n_layers, n_experts) bool tensor. True = pruned.

    Returns: list of hooks for cleanup.
    """
    hooks = []
    n_layers = prune_mask.shape[0]

    for l in range(n_layers):
        if not prune_mask[l].any():
            continue
        block = model.model.layers[l].mlp
        gate = block.gate
        dev = gate.weight.device
        m = prune_mask[l].to(dev)
        # top_k / norm_topk_prob can live on the gate or on the parent MoE
        # block depending on the model class; fall back through both.
        top_k = (getattr(gate, 'top_k', None)
                 or getattr(block, 'top_k', None)
                 or getattr(block, 'num_experts_per_tok', None))
        if top_k is None:
            raise RuntimeError(
                f"Could not determine top_k for layer {l}'s router "
                f"({type(gate).__name__}); add an explicit fallback.")
        norm_topk_prob = (getattr(gate, 'norm_topk_prob', None)
                          if hasattr(gate, 'norm_topk_prob')
                          else getattr(block, 'norm_topk_prob', False))

        def make_hook(mask, tk, norm_tk):
            def hook_fn(module, input, output):
                # zero pruned experts in the gate output so topk cannot
                # select them. The gate may return either the (full_softmax,
                # topk_weights, topk_indices) triple or raw logits.
                if isinstance(output, (tuple, list)):
                    full_softmax = output[0].clone()
                    full_softmax[:, mask] = 0.0
                    router_top_value, router_indices = full_softmax.topk(tk, dim=-1)
                    if norm_tk:
                        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                    router_top_value = router_top_value.to(full_softmax.dtype)
                    return (full_softmax, router_top_value, router_indices)
                else:
                    out = output.clone()
                    out[:, mask] = float('-inf')
                    return out
            return hook_fn

        h = gate.register_forward_hook(make_hook(m, top_k, norm_topk_prob))
        hooks.append(h)

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


@torch.no_grad()
def evaluate_with_mask(model, prune_mask, cal_data, ref_log_probs, ref_masks,
                       batch_size=4):
    """Evaluate KL and ESAP with proper hard pruning (answer tokens only)."""
    hooks = apply_hard_pruning_mask(model, prune_mask)
    device = get_input_device(model)

    total_kl = 0.0
    total_esap = 0.0
    n_answer_tok = 0

    for i in range(0, cal_data.size(0), batch_size):
        batch = cal_data[i:i + batch_size].to(device)
        bi = i // batch_size
        bi = min(bi, len(ref_log_probs) - 1)
        ref_lp = ref_log_probs[bi].to(device)
        # ref_masks is (B, T) unshifted; align with logits[:, :-1, :].
        m = ref_masks[bi][:, 1:].to(device)

        logits = model(input_ids=batch).logits[:, :-1, :].contiguous().float()
        q_lp = F.log_softmax(logits, dim=-1)
        p = ref_lp.float().exp()

        kl = (p * (ref_lp.float() - q_lp)).sum(dim=-1)  # (B, T-1)
        total_kl += (kl * m).sum().item()

        esap = torch.min(p, q_lp.exp()).sum(dim=-1)  # (B, T-1)
        total_esap += (esap * m).sum().item()

        n_answer_tok += m.sum().item()

    remove_hooks(hooks)
    return total_kl / max(n_answer_tok, 1), total_esap / max(n_answer_tok, 1)


# ----------------------------------------------------------------------------
# Frequency / router-weight baselines
# ----------------------------------------------------------------------------


@torch.no_grad()
def compute_frequency(model, data, n_layers, n_experts, top_k, batch_size=4):
    """Compute routing frequency for each expert."""
    device = get_input_device(model)
    freq = torch.zeros(n_layers, n_experts)
    hooks = []

    for l in range(n_layers):
        def make_hook(li, tk):
            def hook_fn(module, input, output):
                # gate returns (full_softmax, topk_weights, topk_indices)
                if isinstance(output, (tuple, list)) and len(output) >= 3:
                    selected = output[2]  # (batch*seq, top_k) indices
                elif isinstance(output, (tuple, list)) and len(output) >= 2:
                    weights = output[0]  # full softmax
                    selected = weights.topk(tk, dim=-1).indices
                else:
                    return output
                counts = torch.zeros(n_experts, device=selected.device)
                for k in range(tk):
                    counts.scatter_add_(
                        0, selected[:, k],
                        torch.ones(selected.shape[0], device=selected.device))
                freq[li] += counts.cpu()
                return output
            return hook_fn
        h = model.model.layers[l].mlp.gate.register_forward_hook(make_hook(l, top_k))
        hooks.append(h)

    for i in tqdm(range(0, data.size(0), batch_size), desc="Routing freq"):
        batch = data[i:i + batch_size].to(device)
        model(batch)

    for h in hooks:
        h.remove()

    return freq


@torch.no_grad()
def compute_router_scores(model, data, n_layers, n_experts, batch_size=4):
    """Sum of router gate weights per expert across the calibration set.

    For each token x the router emits a non-negative gate vector g(x); this
    function accumulates sum_x g_j(x) per expert j. Experts that the router
    consistently assigns high weight to score high. Used as a prior for
    init_alpha_from_router_scores.
    """
    logger.info("Computing router gate-weight sums...")
    device = get_input_device(model)
    scores = torch.zeros(n_layers, n_experts)
    hooks = []

    for l in range(n_layers):
        def make_hook(li):
            def hook_fn(module, input, output):
                # output[0] is the full per-expert softmax over the batch.
                if isinstance(output, (tuple, list)) and len(output) >= 1:
                    full_weights = output[0]
                else:
                    return output
                scores[li] += full_weights.sum(dim=0).cpu()
                return output
            return hook_fn
        h = model.model.layers[l].mlp.gate.register_forward_hook(make_hook(l))
        hooks.append(h)

    for i in tqdm(range(0, data.size(0), batch_size), desc="Router scores"):
        batch = data[i:i + batch_size].to(device)
        model(batch)

    for h in hooks:
        h.remove()

    for l in range(n_layers):
        s = scores[l]
        top5 = s.topk(5)
        bot5_vals, bot5_idx = s.topk(5, largest=False)
        logger.info(f"  Layer {l:2d}: mean={s.mean():.1f} "
                    f"top=[{', '.join(f'e{top5.indices[i]}={top5.values[i]:.0f}' for i in range(3))}] "
                    f"bot=[{', '.join(f'e{bot5_idx[i]}={bot5_vals[i]:.0f}' for i in range(3))}]")

    return scores


def init_alpha_from_router_scores(alpha, router_scores, target, n_layers, n_experts,
                                  spread=3.0):
    """Initialize alpha from router gate-weight sums.

    High router score = router consistently assigns weight to this expert,
    so it gets a low prune probability. Low router score = router rarely
    uses this expert, so it gets a high prune probability.

    Args:
        spread: Controls how strongly the router-score ranking influences
                the init. Higher = stronger prior, stays closer to the
                router-ranked order; lower = weaker prior, optimizer is
                freer to deviate.
    """
    N = n_layers * n_experts
    # normalize per layer to [0, 1] so 1.0 marks the most important expert.
    scores = router_scores.clone()
    for l in range(n_layers):
        s = scores[l]
        smin, smax = s.min(), s.max()
        if smax > smin:
            scores[l] = (s - smin) / (smax - smin)
        else:
            scores[l] = 0.5

    scores_flat = scores.view(N)

    frac = target / N
    base_c = math.log(frac / (1.0 - frac + 1e-10))

    # higher score -> more negative offset -> less prunable.
    offset = spread * (0.5 - scores_flat)

    with torch.no_grad():
        alpha[:, 0] = 0.0
        alpha[:, 1] = base_c + offset.to(alpha.device)

    costs = torch.tensor([0.0, 1.0], device=alpha.device)
    retraction(alpha, costs, target)
    actual = get_expected_budget(alpha, costs)

    prune_probs = torch.softmax(alpha, dim=1)[:, 1]
    pp = prune_probs.view(n_layers, n_experts)
    per_layer = pp.sum(1)
    pl_str = " ".join(f"{per_layer[l].item():.1f}" for l in range(n_layers))
    logger.info(f"Alpha init from router scores: E[budget]={actual:.2f} (target={target})")
    logger.info(f"  Per-layer E[prune]: [{pl_str}]")
    logger.info(f"  Prune prob range: "
                f"[{prune_probs.min():.3f}, {prune_probs.max():.3f}]")


def frequency_uniform_mask(freq, n_layers, n_experts, target_budget):
    """Prune bottom-r experts per layer by frequency, uniformly."""
    r = target_budget // n_layers
    order = freq.argsort(dim=1)
    mask = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    for l in range(n_layers):
        mask[l, order[l, :r]] = True
    return mask


def gradient_sanity_check(model, alpha, shared_state, n_layers, n_experts,
                          cal_data, ref_log_probs, ref_masks, batch_size, budget,
                          kl_topk=0):
    """Verify gradients flow from the KL loss through STE masks to alpha.

    Runs a single forward + backward at tau=1 before the main optimization
    starts. A None gradient or a zero gradient on alpha means the STE
    wrappers did not actually wire the routing decision into autograd, and
    the search will silently fail. Catching that here is much cheaper than
    debugging an entire run.
    """
    logger.info("Gradient sanity check...")

    alpha.grad = None
    ste_keep, _ = compute_ste_masks(alpha, tau=1.0, budget=budget)
    shared_state['ste_masks'] = ste_keep.view(n_layers, n_experts)

    device = get_input_device(model)
    batch = cal_data[:batch_size].to(device)
    ref_lp = ref_log_probs[0]
    ref_mask = ref_masks[0]

    loss = compute_kl_loss(model, batch, ref_lp, mask=ref_mask, topk=kl_topk)
    loss.backward()

    if alpha.grad is None:
        raise RuntimeError("alpha.grad is None after backward; gradient flow "
                           "from the model output back to alpha is broken.")

    gn = alpha.grad.norm().item()
    nz = (alpha.grad.abs() > 1e-12).sum().item()
    logger.info(f"Gradient check OK: norm={gn:.6f}, "
                f"nonzero={nz}/{alpha.numel()}")
    alpha.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Optimization loop
# ----------------------------------------------------------------------------


def optimize(model, cal_data, ref_log_probs, ref_masks,
             n_layers, n_experts, target_budget,
             n_steps=200, lr=0.1, tau_init=1.0, tau_min=0.05,
             n_gumbel_samples=4, batch_size=4,
             log_interval=5, kl_topk=0,
             entropy_reg=0.0, antithetic=False,
             router_scores=None, router_spread=3.0,
             per_layer_budget=False):
    """
    Gumbel-STE optimization for per-expert pruning decisions.

    Each step:
      1. For each Gumbel sample:
         a. Sample Gumbel noise, compute budget-feasible hard assignment.
         b. Build STE masks (hard forward, soft backward).
         c. Forward through patched MoE model -> KL loss.
         d. Backward -> gradients flow through STE to alpha.
      2. Riemannian projection: remove budget-changing gradient component.
      3. Adam step in the tangent space.
      4. Budget retraction: binary-search shift to restore E[budget]=target.
      5. Vector transport: reproject Adam moments onto new tangent plane.
    """
    N = n_layers * n_experts
    ctrl = torch.device('cuda:0')
    costs = torch.tensor([0.0, 1.0], device=ctrl)

    if per_layer_budget:
        assert target_budget % n_layers == 0, (
            f"per-layer mode: target_budget ({target_budget}) must divide "
            f"n_layers ({n_layers})")
        target_per_layer = target_budget // n_layers
        assert 0 < target_per_layer < n_experts, \
            f"per-layer budget {target_per_layer} must be in (0, {n_experts})"
        logger.info(f"PER-LAYER mode: prune {target_per_layer}/{n_experts} per layer "
                    f"({n_layers} layers x {target_per_layer} = {target_budget} total)")

    # alpha: (N, 2) logits, column 0 keeps, column 1 prunes.
    alpha = torch.zeros(N, 2, dtype=torch.float32, device=ctrl,
                        requires_grad=True)
    if router_scores is not None:
        init_alpha_from_router_scores(alpha, router_scores, target_budget,
                                      n_layers, n_experts, spread=router_spread)
    else:
        init_alpha(alpha, target_budget, N)

    optimizer = torch.optim.Adam([alpha], lr=lr)

    model.requires_grad_(False)
    model.eval()

    shared_state = {'ste_masks': None}
    wrappers = install_wrappers(model, n_layers, n_experts, shared_state)

    gradient_sanity_check(model, alpha, shared_state, n_layers, n_experts,
                          cal_data, ref_log_probs, ref_masks, batch_size,
                          target_budget, kl_topk=kl_topk)

    input_device = get_input_device(model)
    all_bi = list(range(0, cal_data.size(0), batch_size))
    n_total_batches = len(all_bi)

    logger.info(f"Gumbel-STE optimization: {n_steps} steps, lr={lr}, "
                f"tau {tau_init}->{tau_min}")
    logger.info(f"Budget: {target_budget}/{N} experts pruned "
                f"({n_layers} layers x {n_experts} experts)")
    logger.info(f"Gumbel samples/step: {n_gumbel_samples}, "
                f"each samples 1 random batch from {n_total_batches} "
                f"(batch_size={batch_size}, total seqs={cal_data.size(0)})")

    history = []
    t0 = time.time()

    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)

        optimizer.zero_grad()
        total_loss = 0.0
        # antithetic sampling pairs each Gumbel noise with its negation
        # for variance reduction, so we draw half as many base samples.
        n_base = n_gumbel_samples // 2 if antithetic else n_gumbel_samples
        n_base = max(n_base, 1)
        n_evals = n_base * 2 if antithetic else n_base
        scale = 1.0 / n_evals

        for _gs in range(n_base):
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(alpha).clamp(1e-20)) + 1e-20)

            noise_variants = [gumbel_noise]
            if antithetic:
                noise_variants.append(-gumbel_noise)

            for gn_var in noise_variants:
                bi = all_bi[torch.randint(n_total_batches, (1,)).item()]

                noisy = (alpha + gn_var) / tau
                soft_probs = F.softmax(noisy, dim=1)
                soft_keep = soft_probs[:, 0]

                if per_layer_budget:
                    hard_asgn = budget_assignment_per_layer(
                        noisy.detach(), target_per_layer, n_layers, n_experts)
                else:
                    hard_asgn = budget_assignment(noisy.detach(), target_budget)
                hard_keep = 1.0 - hard_asgn.float()

                ste_keep = hard_keep + (soft_keep - soft_keep.detach())
                shared_state['ste_masks'] = ste_keep.view(n_layers, n_experts)

                batch = cal_data[bi:bi + batch_size].to(input_device)
                ref_idx = min(bi // batch_size, len(ref_log_probs) - 1)
                ref_lp = ref_log_probs[ref_idx]
                ref_mask = ref_masks[ref_idx]

                loss = compute_kl_loss(model, batch, ref_lp, mask=ref_mask,
                                       topk=kl_topk)
                (loss * scale).backward()
                total_loss += loss.item()

        avg_loss = total_loss / n_evals

        # entropy regularizer pushes alphas away from indecisive (high entropy)
        # configurations and toward 0/1 keep/prune decisions.
        if entropy_reg > 0:
            probs_for_ent = torch.softmax(alpha, dim=1)
            ent_loss = -(probs_for_ent * (probs_for_ent + 1e-10).log()).sum(1).mean()
            (entropy_reg * ent_loss).backward()

        # Riemannian Adam: project grad onto the budget tangent plane(s),
        # take an Adam step, retract, then transport the first moment.
        if per_layer_budget:
            proj_coeff = project_gradient_per_layer(
                alpha, costs, n_layers, n_experts)
        else:
            proj_coeff, _, _ = project_gradient(alpha, costs)
        gn = alpha.grad.norm().item()

        optimizer.step()

        if per_layer_budget:
            bud = retraction_per_layer(
                alpha, costs, target_per_layer, n_layers, n_experts)
            bud = bud * n_layers
        else:
            bud = retraction(alpha, costs, target_budget)

        if per_layer_budget:
            vector_transport_per_layer(optimizer, alpha, costs, n_layers, n_experts)
        else:
            vector_transport(optimizer, alpha, costs)

        with torch.no_grad():
            probs = torch.softmax(alpha, dim=1)
            prune_probs = probs[:, 1]
            ent = -(probs * (probs + 1e-10).log()).sum(1).mean().item()
            decided = (probs.max(1).values > 0.9).sum().item()
            n_hard = (probs.max(1).values > 0.99).sum().item()
            per_layer = prune_probs.view(n_layers, n_experts).sum(1)

        history.append(dict(
            step=step, loss=avg_loss, budget=bud, tau=tau,
            grad_norm=gn, entropy=ent, decided=decided, n_hard=n_hard,
            proj_coeff=proj_coeff))

        if step % log_interval == 0 or step == n_steps - 1:
            el = time.time() - t0
            eta = (el / (step + 1)) * (n_steps - step - 1)
            em, es = divmod(int(eta), 60)
            pl = " ".join(f"{per_layer[l].item():.1f}" for l in range(n_layers))
            logger.info(
                f"[{step:4d}] KL={avg_loss:.4f} bud={bud:.1f} "
                f"tau={tau:.3f} ent={ent:.3f} "
                f"dec={decided}/{N} hard={n_hard} "
                f"grad={gn:.4f} proj={proj_coeff:.3f} "
                f"E[r]=[{pl}]  ETA {em}m{es:02d}s")

        torch.cuda.empty_cache()

    remove_wrappers(wrappers)

    # final deterministic assignment (no Gumbel noise)
    if per_layer_budget:
        final_asgn = budget_assignment_per_layer(
            alpha.detach(), target_per_layer, n_layers, n_experts)
    else:
        final_asgn = budget_assignment(alpha.detach(), target_budget)
    final_mask = final_asgn.view(n_layers, n_experts).bool()

    return alpha, final_mask, history


