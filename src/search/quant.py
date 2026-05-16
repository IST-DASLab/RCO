"""RCO mixed-precision quantization search.

Each step samples a Gumbel-perturbed bitwidth assignment, runs a forward
pass through an interpolated model, projects the loss gradient onto the
budget tangent plane, takes an Adam step, and binary-searches a scalar
shift to restore the average-bitwidth budget.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    get_input_device,
    get_layer_weights,
    resolve_module,
    set_layer_weights,
)
from manifold import project_gradient, retraction, vector_transport
from metrics import (
    compute_all_metrics,
    compute_ce_loss,
    compute_kl_loss,
    compute_objective,
    compute_reference_log_probs,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def parse_bitwidth_map(bitwidth_map_str: str) -> Dict[int, float]:
    if not bitwidth_map_str:
        return {}
    mapping = {}
    for pair in bitwidth_map_str.split(','):
        pair = pair.strip()
        if ':' in pair:
            nominal, actual = pair.split(':')
            mapping[int(nominal.strip())] = float(actual.strip())
    return mapping


def get_actual_bitwidth(nominal: int, bitwidth_map: Dict[int, float]) -> float:
    return bitwidth_map.get(nominal, float(nominal))


def parse_bitwidths(bitwidths_str: str) -> List[int]:
    return sorted(int(b.strip()) for b in bitwidths_str.split(','))


def build_ref_batches(calibration_data, ref_log_probs, batch_size):
    if ref_log_probs is None:
        return None
    n_samples = calibration_data.size(0)
    ref_batches = []
    for i in range(0, n_samples, batch_size):
        batch_idx = i // batch_size
        ref_batches.append(ref_log_probs[min(batch_idx, len(ref_log_probs) - 1)])
    return ref_batches


# ----------------------------------------------------------------------------
# Interpolated model
# ----------------------------------------------------------------------------


class WeightInterpolation(nn.Module):
    """
    Softmax interpolation across pre-quantized weight versions.
    W_mixed = W_ref + sum_{b != ref} probs[b] * delta_b
    """
    def __init__(self, deltas: List[torch.Tensor], alpha: torch.Tensor,
                 group_idx: int, shared_state: dict = None,
                 cpu_deltas: bool = False):
        super().__init__()
        self.n_deltas = len(deltas)
        self.cpu_deltas = cpu_deltas
        if cpu_deltas:
            # plain attribute so register_parametrization's .to() does not move it.
            self._delta_list = [d.cpu() for d in deltas]
        else:
            for i, d in enumerate(deltas):
                self.register_buffer(f'delta_{i}', d)
        self.alpha = alpha
        self.group_idx = group_idx
        self._shared = shared_state or {}

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        override = self._shared.get('probs_override')
        if override is not None:
            probs = override[self.group_idx].to(W.device)
        else:
            probs = torch.softmax(self.alpha[self.group_idx].to(W.device), dim=0)
        result = W
        for i in range(self.n_deltas):
            if self.cpu_deltas:
                delta = self._delta_list[i].to(
                    device=W.device, dtype=W.dtype, non_blocking=True
                )
            else:
                delta = getattr(self, f'delta_{i}').to(W.dtype)
            result = result + probs[i] * delta
        return result


class InterpolatedModel:
    """
    Manages differentiable weight interpolation across bitwidths.
    alpha: (n_groups, n_bw) logits. Last bitwidth is reference (highest).

    The alpha tensor and projection math live on a single control device
    (default cuda:0). Model layers may be spread across multiple GPUs via
    device_map='auto'. The WeightInterpolation parametrizations handle
    cross-device transfers internally.
    """

    CONTROL_DEVICE = torch.device('cuda:0')

    def __init__(self, model, weight_store, bitwidths, layer_names,
                 layer_groups, param_counts, bitwidth_map=None,
                 cpu_deltas=False):
        self.model = model
        self.weight_store = weight_store
        self.bitwidths = sorted(bitwidths)
        self.n_bw = len(self.bitwidths)
        self.ref_bw = self.bitwidths[-1]
        self.non_ref_bws = self.bitwidths[:-1]
        self.layer_names = layer_names
        self.groups = layer_groups
        self.param_counts = param_counts
        self.n_groups = len(self.groups)
        self.input_device = get_input_device(model)

        if bitwidth_map is None:
            bitwidth_map = {}
        self.bitwidth_map = bitwidth_map

        self.actual_bits = torch.tensor(
            [get_actual_bitwidth(bw, bitwidth_map) for bw in self.bitwidths],
            dtype=torch.float32, device=self.CONTROL_DEVICE
        )
        self.layer_to_group = {}
        for i, g in enumerate(self.groups):
            for name in g.layer_names:
                self.layer_to_group[name] = i

        self.alpha = torch.zeros(
            self.n_groups, self.n_bw,
            dtype=torch.float32, device=self.CONTROL_DEVICE, requires_grad=True
        )

        total_params = sum(param_counts.get(n, 0) for n in layer_names)
        self.group_param_fracs = torch.zeros(
            self.n_groups, dtype=torch.float32, device=self.CONTROL_DEVICE
        )
        for i, g in enumerate(self.groups):
            g_params = sum(param_counts.get(n, 0) for n in g.layer_names)
            self.group_param_fracs[i] = g_params / total_params if total_params > 0 else 0

        self._shared_state = {'probs_override': None}
        self.cpu_deltas = cpu_deltas

    def setup(self):
        logger.info(f"Setting up interpolated model ({self.n_bw} bitwidths)...")
        logger.info(f"  Control device: {self.CONTROL_DEVICE}")
        logger.info(f"  Input device:   {self.input_device}")
        self.model.requires_grad_(False)

        for name in tqdm(self.layer_names, desc="Preparing layers"):
            W_ref = self.weight_store.get_layer_weight(name, self.ref_bw)
            set_layer_weights(self.model, name, W_ref)

            module = resolve_module(self.model, name)
            device = module.weight.device

            deltas = []
            delta_device = torch.device('cpu') if self.cpu_deltas else device
            for bw in self.non_ref_bws:
                if bw == 0:  # prune
                    delta = (-W_ref.float()).to(dtype=torch.bfloat16, device=delta_device)
                else:
                    W_bw = self.weight_store.get_layer_weight(name, bw)
                    delta = (W_bw.float() - W_ref.float()).to(dtype=torch.bfloat16, device=delta_device)
                deltas.append(delta)

            group_idx = self.layer_to_group[name]
            parametrization = WeightInterpolation(
                deltas, self.alpha, group_idx, self._shared_state,
                cpu_deltas=self.cpu_deltas,
            )
            torch.nn.utils.parametrize.register_parametrization(
                module, 'weight', parametrization
            )

        devices_seen = set()
        for name in self.layer_names:
            module = resolve_module(self.model, name)
            devices_seen.add(str(module.weight.device))
        logger.info(f"Ready: {self.n_groups} groups, {len(self.layer_names)} layers, "
                    f"{self.n_bw} bitwidths (ref={self.ref_bw})")
        logger.info(f"Layer devices: {sorted(devices_seen)}")

        # the sanity check backprops through the model; skip for cpu_deltas /
        # REINFORCE which do not need autograd to flow through alpha.
        if not self.cpu_deltas:
            self._gradient_sanity_check()
        else:
            logger.info("Skipping gradient sanity check (cpu_deltas mode)")

    def _gradient_sanity_check(self):
        logger.info("Gradient sanity check...")
        gc_attr = getattr(self.model, 'gradient_checkpointing', None)
        gc_method = getattr(self.model, 'is_gradient_checkpointing', None)
        logger.info(f"  gradient_checkpointing attr: {gc_attr}, "
                    f"is_gradient_checkpointing: {gc_method}")
        try:
            inner = self.model.model.layers[0]
            logger.info(f"  layer[0].gradient_checkpointing: "
                        f"{getattr(inner, 'gradient_checkpointing', None)}")
        except Exception as e:
            logger.info(f"  inner check failed: {e}")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"  cuda:{i} pre-check: allocated={alloc:.2f}GB reserved={reserved:.2f}GB")
        self.alpha.grad = None
        dummy_ids = torch.tensor([[1, 2, 3, 4]], device=self.input_device)
        with torch.enable_grad():
            out = self.model(input_ids=dummy_ids)
            loss = out.logits.sum()
            loss.backward()

        if self.alpha.grad is None:
            raise RuntimeError("alpha.grad is None after backward")

        grad_norm = self.alpha.grad.norm().item()
        grad_nonzero = (self.alpha.grad.abs() > 1e-12).sum().item()
        logger.info(f"Gradient check OK: norm={grad_norm:.6f}, "
                    f"nonzero={grad_nonzero}/{self.n_groups * self.n_bw}")
        self.alpha.grad = None
        torch.cuda.empty_cache()

    def set_probs_override(self, probs):
        self._shared_state['probs_override'] = probs

    def clear_probs_override(self):
        self._shared_state['probs_override'] = None

    def sample_gumbel_softmax(self, tau, hard=True):
        return F.gumbel_softmax(self.alpha, tau=tau, hard=hard, dim=1)

    def get_avg_bits(self):
        with torch.no_grad():
            probs = torch.softmax(self.alpha, dim=1)
            per_group = (probs * self.actual_bits.unsqueeze(0)).sum(dim=1)
            return (per_group * self.group_param_fracs).sum().item()

    def init_alpha_from_sensitivity(self, target_bits):
        """Initialize alpha using delta norms as sensitivity proxy."""
        distortion = torch.zeros(self.n_groups, self.n_bw,
                                device=self.CONTROL_DEVICE)

        for name in self.layer_names:
            module = resolve_module(self.model, name)
            group_idx = self.layer_to_group[name]
            n_params = self.param_counts.get(name, 1)

            param = module.parametrizations.weight[0]
            for i in range(param.n_deltas):
                if param.cpu_deltas:
                    delta = param._delta_list[i]
                else:
                    delta = getattr(param, f'delta_{i}')
                # normalized MSE keeps the scale comparable across layers.
                distortion[group_idx, i] += (delta.float().pow(2).sum() / n_params).item()

            distortion[group_idx, -1] += 0.0

        # lower distortion -> higher logit; epsilon avoids log(0).
        alpha_init = -torch.log(distortion + 1e-8)
        alpha_init = alpha_init - alpha_init.mean(dim=1, keepdim=True)

        with torch.no_grad():
            self.alpha.copy_(alpha_init)

        self.retract(target_bits)
        actual = self.get_avg_bits()
        logger.info(f"Sensitivity init: E[bits]={actual:.4f} (target={target_bits})")

    def init_alpha_to_bits(self, target_bits):
        """Binary search for alpha init so E[bits] = target from step 0."""
        bits = self.actual_bits.cpu().float()
        lo, hi = -10.0, 10.0

        def expected_bits(beta):
            logits = -beta * bits
            probs = torch.softmax(logits, dim=0)
            return (probs * bits).sum().item()

        for _ in range(100):
            mid = (lo + hi) / 2.0
            if expected_bits(mid) > target_bits:
                lo = mid
            else:
                hi = mid

        beta = (lo + hi) / 2.0
        init_logits = -beta * bits
        with torch.no_grad():
            self.alpha.copy_(
                init_logits.unsqueeze(0).expand(self.n_groups, -1).to(self.CONTROL_DEVICE)
            )

        actual = self.get_avg_bits()
        logger.info(f"Alpha init: beta={beta:.4f}, E[bits]={actual:.4f} (target={target_bits})")

    def retract(self, target_bits):
        """Retract alpha so the weighted-average bitwidth equals target."""
        return retraction(self.alpha, self.actual_bits, target_bits,
                          weights=self.group_param_fracs, tol=1e-4)

    def get_probs(self):
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=1)

    def get_prob_dict(self):
        probs = self.get_probs()
        result = {}
        for i, g in enumerate(self.groups):
            p = {bw: probs[i, j].item() for j, bw in enumerate(self.bitwidths)}
            for name in g.layer_names:
                result[name] = p
        return result

    def cleanup(self):
        for name in self.layer_names:
            module = resolve_module(self.model, name)
            torch.nn.utils.parametrize.remove_parametrizations(
                module, 'weight', leave_parametrized=False
            )
        self.model.requires_grad_(True)
        logger.info("Interpolated model cleaned up")


# ----------------------------------------------------------------------------
# Budget-constrained discrete assignment
# ----------------------------------------------------------------------------


def round_with_budget_dp(prob_dict, param_counts, bitwidths, target_avg_bits,
                         groups, bitwidth_map=None, resolution=10000):
    """Optimal discrete assignment via dynamic programming."""
    if bitwidth_map is None:
        bitwidth_map = {}

    bitwidths_sorted = sorted(bitwidths)
    total_params = sum(param_counts.get(n, 0) for n in prob_dict)
    if total_params == 0:
        return {n: bitwidths_sorted[0] for n in prob_dict}

    budget = int(round(target_avg_bits * resolution))

    group_info = []
    for g in groups:
        rep = g.layer_names[0]
        probs = prob_dict[rep]
        g_params = sum(param_counts.get(n, 0) for n in g.layer_names)
        frac = g_params / total_params

        options = []
        for bw in bitwidths_sorted:
            actual = get_actual_bitwidth(bw, bitwidth_map)
            cost = int(round(actual * frac * resolution))
            # log-prob is the value: higher prob = optimizer preferred this option.
            value = math.log(max(probs.get(bw, 1e-10), 1e-10))
            options.append((bw, cost, value))
        group_info.append((g, options))

    n_groups = len(group_info)

    # dp[j] = max total value achievable using exactly j budget units.
    NEG_INF = float('-inf')
    dp = [NEG_INF] * (budget + 1)
    dp[0] = 0.0
    choice = [[None] * (budget + 1) for _ in range(n_groups)]

    for i, (g, options) in enumerate(group_info):
        new_dp = [NEG_INF] * (budget + 1)
        for bw, cost, value in options:
            for j in range(budget + 1):
                prev = j - cost
                if 0 <= prev <= budget and dp[prev] > NEG_INF:
                    candidate = dp[prev] + value
                    if candidate > new_dp[j]:
                        new_dp[j] = candidate
                        choice[i][j] = (bw, prev)
        dp = new_dp

    # find best feasible budget, allowing small slack on either side.
    best_j = budget
    if dp[best_j] == NEG_INF:
        for delta in range(1, budget):
            for j in [budget - delta, budget + delta]:
                if 0 <= j <= budget and dp[j] > NEG_INF:
                    best_j = j
                    break
            if dp[best_j] > NEG_INF:
                break

    assignment = {}
    j = best_j
    for i in range(n_groups - 1, -1, -1):
        bw, prev_j = choice[i][j]
        for name in group_info[i][0].layer_names:
            assignment[name] = bw
        j = prev_j

    return assignment


def budget_constrained_argmax(noisy_logits, group_param_fracs, actual_bits, 
                               target_bits, resolution=500):
    """Solve knapsack: best discrete assignment under budget."""
    n_groups, n_bw = noisy_logits.shape
    budget = int(round(target_bits * resolution))
    
    logits_np = noisy_logits.detach().cpu().numpy()
    fracs = group_param_fracs.detach().cpu().numpy()
    bits_np = actual_bits.detach().cpu().numpy()

    # costs[g][b] = discretized budget cost of assigning group g to bitwidth b.
    costs = np.round(np.outer(fracs, bits_np) * resolution).astype(int)
    
    NEG_INF = -1e30
    dp = np.full(budget + 1, NEG_INF)
    dp[0] = 0.0
    backtrack = np.zeros((n_groups, budget + 1), dtype=int)
    
    for g in range(n_groups):
        new_dp = np.full(budget + 1, NEG_INF)
        for b in range(n_bw):
            c = costs[g, b]
            v = logits_np[g, b]
            for j in range(c, budget + 1):
                candidate = dp[j - c] + v
                if candidate > new_dp[j]:
                    new_dp[j] = candidate
                    backtrack[g, j] = b
        dp = new_dp

    best_j = np.argmax(dp[:budget + 1])

    assignment = np.zeros(n_groups, dtype=int)
    j = best_j
    for g in range(n_groups - 1, -1, -1):
        assignment[g] = backtrack[g, j]
        j -= costs[g, assignment[g]]
    
    return torch.tensor(assignment, device=noisy_logits.device)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def run_sensitivity_probe(interp, calibration_data, ref_batches, batch_size,
                          objective, masks, input_device):
    """
    Compute loss with all-min-bits and all-max-bits to verify the loss landscape
    has signal. Logs the gap -- if it's near zero, the optimizer has nothing to learn.
    """
    logger.info("[SensitivityProbe] Running loss probe: all-min-bits vs all-max-bits...")
    n_bw = interp.n_bw
    n_groups = interp.n_groups

    def eval_fixed(bw_idx):
        probs_override = torch.zeros(n_groups, n_bw, device=interp.CONTROL_DEVICE)
        probs_override[:, bw_idx] = 1.0
        interp.set_probs_override(probs_override)
        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, calibration_data.size(0), batch_size):
                batch = calibration_data[i:i + batch_size].to(input_device)
                batch_masks = masks[i:i + batch_size].to(input_device) if masks is not None else None
                ref_lp = ref_batches[i // batch_size] if ref_batches else None
                loss = compute_objective(interp.model, batch, objective, ref_lp, batch_masks)
                total += loss.item()
                n += 1
        interp.clear_probs_override()
        return total / max(n, 1)

    loss_min = eval_fixed(0)   # lowest bitwidth (index 0)
    loss_max = eval_fixed(-1)  # highest bitwidth (index -1, reference)
    gap = loss_min - loss_max
    logger.info(
        f"[SensitivityProbe] all-{interp.bitwidths[0]}bit loss={loss_min:.6f}  "
        f"all-{interp.bitwidths[-1]}bit loss={loss_max:.6f}  "
        f"gap={gap:.6f}  ({'OK: signal present' if gap > 1e-5 else 'WARN: gap very small'})"
    )
    return loss_min, loss_max


def log_gradient_diagnostics(interp, step, proj_coeff, raw_norm, proj_norm,
                              budget_before_correction, budget_after_correction,
                              alpha_snapshot_prev):
    """
    Log detailed gradient and alpha diagnostics:
    - Per-bitwidth mean gradient (sign tells us which direction optimizer pushes)
    - Projection coefficient and fraction of gradient removed
    - Budget drift before/after correction
    - Alpha movement since last diagnostic call
    """
    with torch.no_grad():
        grad = interp.alpha.grad
        probs = torch.softmax(interp.alpha, dim=1)

        # grad_per_bw[b] > 0 means Adam decreases alpha for bitwidth b
        # (pushes away); < 0 means Adam pushes toward it.
        grad_per_bw = grad.mean(dim=0)
        prob_per_bw = probs.mean(dim=0)

        budget_drift = abs(budget_before_correction - budget_after_correction)

        if alpha_snapshot_prev is not None:
            alpha_delta = (interp.alpha - alpha_snapshot_prev).abs()
            alpha_move = alpha_delta.mean().item()
            alpha_move_max = alpha_delta.max().item()
        else:
            alpha_move = 0.0
            alpha_move_max = 0.0

        frac_removed = 1.0 - (proj_norm / (raw_norm + 1e-12))

        logger.info(f"[GradDiag step={step}]")
        logger.info(f"  grad: raw_norm={raw_norm:.5f}  proj_norm={proj_norm:.5f}  "
                    f"proj_coeff={proj_coeff:.4f}  frac_removed={frac_removed:.3f}")
        logger.info(f"  budget: before_correction={budget_before_correction:.5f}  "
                    f"after={budget_after_correction:.5f}  drift={budget_drift:.6f}  "
                    f"({'OK' if budget_drift < 0.02 else 'LARGE: check'})")

        bw_lines = []
        for j, bw in enumerate(interp.bitwidths):
            g_j = grad_per_bw[j].item()
            p_j = prob_per_bw[j].item()
            direction = "down" if g_j > 0 else "up"
            bw_lines.append(f"{bw}b: grad_mean={g_j:+.5f}{direction} prob={p_j:.3f}")
        logger.info("  per-bw (grad>0->Adam pushes away, grad<0->pushes toward):")
        for line in bw_lines:
            logger.info(f"    {line}")

        logger.info(f"  alpha movement (since last diag): mean={alpha_move:.5f}  max={alpha_move_max:.5f}")

        argmax_idx = probs.argmax(dim=1)
        bw_dist = {bw: (argmax_idx == j).sum().item() for j, bw in enumerate(interp.bitwidths)}
        bw_dist_str = " ".join(f"{bw}b:{c}" for bw, c in sorted(bw_dist.items()))
        logger.info(f"  argmax distribution: [{bw_dist_str}]")


# ----------------------------------------------------------------------------
# Optimization loops
# ----------------------------------------------------------------------------


def optimize_projected_gumbel(
    interp: InterpolatedModel,
    calibration_data: torch.Tensor,
    target_avg_bits: float,
    n_steps: int = 400,
    lr: float = 0.05,
    tau_init: float = 1.0,
    tau_min: float = 0.05,
    n_gumbel_samples: int = 1,
    batch_size: int = 4,
    batches_per_step: int = 0,
    masks: Optional[torch.Tensor] = None,
    log_interval: int = 1,
    objective: str = 'kl',
    ref_log_probs: Optional[List[torch.Tensor]] = None,
    grad_debug: bool = False,
    grad_debug_interval: int = 10,
) -> Tuple[Dict[str, Dict[int, float]], List[dict]]:
    """
    Optimize bitwidth assignment via projected Gumbel-softmax.

    Each step:
      1. Draw Gumbel-softmax sample (hard=True, straight-through estimator)
      2. Forward pass -> loss, backward -> dL/dalpha
      3. Project gradient onto budget-preserving subspace
      4. Adam step
      5. Correct any residual budget drift

    Temperature anneals from tau_init to tau_min via cosine schedule,
    driving the solution toward discrete one-hot assignments.

    No budget penalty, no Lagrangian, no dual variable.
    """
    if objective == 'kl' and ref_log_probs is None:
        raise ValueError("ref_log_probs required for KL objective")

    input_device = interp.input_device

    optimizer = torch.optim.Adam([interp.alpha], lr=lr)
    n_samples = calibration_data.size(0)
    n_total_batches = (n_samples + batch_size - 1) // batch_size
    history = []

    ref_batches = build_ref_batches(calibration_data, ref_log_probs, batch_size)

    # batches_per_step=0 means use all batches each step.
    all_batch_indices = list(range(0, n_samples, batch_size))
    n_total_batches_actual = len(all_batch_indices)
    use_minibatch = batches_per_step > 0 and batches_per_step < n_total_batches_actual
    if use_minibatch:
        logger.info(f"Mini-batching: {batches_per_step}/{n_total_batches_actual} batches per step")
    else:
        batches_per_step = n_total_batches_actual

    logger.info(f"Projected Gumbel-softmax: tau {tau_init} -> {tau_min}, "
                f"{n_gumbel_samples} sample(s)/step, lr={lr}")
    logger.info(f"Target: {target_avg_bits} bits, {interp.n_groups} groups, "
                f"{interp.n_bw} bitwidths {interp.bitwidths}")
    logger.info(f"No budget penalty. Gradient projection + post-step correction.")
    logger.info(f"Input device: {input_device}")

    if grad_debug:
        logger.info(f"[GradDebug] enabled, diagnostics every {grad_debug_interval} steps")
        run_sensitivity_probe(
            interp, calibration_data, ref_batches, batch_size,
            objective, masks, input_device
        )
        alpha_snapshot = interp.alpha.detach().clone()
    else:
        alpha_snapshot = None

    start_time = time.time()

    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)

        # exponential anneal from tau_init to tau_min.
        tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)

        optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        if use_minibatch:
            batch_indices = [all_batch_indices[i] for i in
                            torch.randperm(n_total_batches_actual)[:batches_per_step].tolist()]
        else:
            batch_indices = all_batch_indices
        n_bi = len(batch_indices)

        # average gradients across Gumbel samples to reduce variance.
        n_evals = n_gumbel_samples

        for _gs in range(n_gumbel_samples):
            u = torch.rand_like(interp.alpha).clamp(min=1e-20, max=1 - 1e-20)
            gumbel_noise = -torch.log(-torch.log(u) + 1e-20)

            for bi, i in enumerate(batch_indices):
                # rebuild gumbel_probs per batch from the same noise so each
                # batch has its own graph (no retain_graph=True needed). The
                # hard assignment is identical across batches in this sample.
                noisy_logits = (interp.alpha + gumbel_noise) / tau
                soft = F.softmax(noisy_logits, dim=-1)

                hard_idx = budget_constrained_argmax(
                    noisy_logits, interp.group_param_fracs,
                    interp.actual_bits, target_avg_bits
                )

                hard_onehot = F.one_hot(hard_idx, interp.n_bw).float()
                gumbel_probs = hard_onehot - soft.detach() + soft  # straight-through

                interp.set_probs_override(gumbel_probs)

                batch = calibration_data[i:i + batch_size].to(input_device)
                batch_masks = (
                    masks[i:i + batch_size].to(input_device)
                    if masks is not None else None
                )
                ref_lp = ref_batches[i // batch_size] if ref_batches else None

                with torch.enable_grad():
                    loss = compute_objective(
                        interp.model, batch, objective, ref_lp, batch_masks
                    )
                    scaled = loss / (n_bi * n_evals)
                    scaled.backward()

                total_loss += loss.item()
                n_batches += 1

        interp.clear_probs_override()
        avg_loss = total_loss / n_batches

        # Riemannian Adam: (1) project gradient onto budget tangent plane so
        # Adam's moments only see budget-neutral signal, (2) Adam step in
        # tangent space, (3) retract to the budget manifold, (4) parallel
        # transport: re-project moments onto the rotated tangent plane.
        proj_coeff, raw_norm, proj_norm = 0.0, 0.0, 0.0
        if interp.alpha.grad is not None:
            proj_coeff, raw_norm, proj_norm = project_gradient(
                interp.alpha, interp.actual_bits, interp.group_param_fracs
            )
        grad_norm = interp.alpha.grad.norm().item() if interp.alpha.grad is not None else 0.0

        optimizer.step()

        budget_before = interp.get_avg_bits()
        corrected_bits = interp.retract(target_avg_bits)

        vector_transport(
            optimizer, interp.alpha, interp.actual_bits, interp.group_param_fracs
        )

        if grad_debug and (step % grad_debug_interval == 0 or step == n_steps - 1):
            log_gradient_diagnostics(
                interp, step, proj_coeff, raw_norm, proj_norm,
                budget_before, corrected_bits, alpha_snapshot
            )
            alpha_snapshot = interp.alpha.detach().clone()

        with torch.no_grad():
            probs = torch.softmax(interp.alpha, dim=1)
            max_probs = probs.max(dim=1).values
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean().item()
            n_decided = (max_probs > 0.9).sum().item()
            n_hard = (max_probs > 0.99).sum().item()
            argmax_indices = probs.argmax(dim=1)
            bw_counts = {bw: (argmax_indices == j).sum().item()
                         for j, bw in enumerate(interp.bitwidths)}

        entry = {
            'step': step,
            'loss': avg_loss,
            'avg_bits': corrected_bits,
            'tau': tau,
            'grad_norm': grad_norm,
            'entropy': entropy,
            'n_decided': n_decided,
            'n_hard': n_hard,
            'bw_counts': bw_counts,
        }
        history.append(entry)

        if step % log_interval == 0 or step == n_steps - 1:
            elapsed = time.time() - start_time
            steps_done = step + 1
            steps_left = n_steps - steps_done
            eta_seconds = (elapsed / steps_done) * steps_left
            eta_min, eta_sec = divmod(int(eta_seconds), 60)

            obj_label = "KL" if objective == 'kl' else "CE"
            bw_str = " ".join(f"{bw}b:{c}" for bw, c in sorted(bw_counts.items()))
            logger.info(
                f"[Step {step:4d}] {obj_label}={avg_loss:.6f}  "
                f"bits={corrected_bits:.4f}  tau={tau:.4f}  "
                f"ent={entropy:.3f}  decided={n_decided}/{interp.n_groups}  "
                f"hard={n_hard}  [{bw_str}]  grad={grad_norm:.4f}  "
                f"ETA={eta_min}m{eta_sec:02d}s"
            )

        torch.cuda.empty_cache()

    return interp.get_prob_dict(), history



# ----------------------------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------------------------


def evaluate_assignment(model, weight_store, assignment, calibration_data,
                        baseline_topk_vals, baseline_topk_idx,
                        batch_size=4, top_k=10, masks=None):
    logger.info("Evaluating final assignment...")
    for name, bw in assignment.items():
        if bw == 0:
            set_layer_weights(model, name, torch.zeros_like(
                get_layer_weights(model, name)
            ))
        else:
            W = weight_store.get_layer_weight(name, bw)
            set_layer_weights(model, name, W)

    with torch.no_grad():
        nll, kl = compute_all_metrics(
            model, calibration_data, baseline_topk_vals, baseline_topk_idx,
            batch_size, top_k, masks=masks
        )

    results = {
        'nll': nll.item() if torch.is_tensor(nll) else nll,
        'kl': kl.item() if torch.is_tensor(kl) else kl,
    }
    logger.info(f"Metrics: NLL={results['nll']:.4f}, KL={results['kl']:.6f}")
    return results

