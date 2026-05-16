"""Budget-manifold operations.

The constraint is the softmax-budget equality (paper Eq. 2):

    C(alpha) = sum_i w_i * sum_k p_ik * c_k = B,
    p_ik     = softmax(alpha_i)_k.

This module exposes the three primitives the paper uses to move alpha
along that manifold (paper Sec. 2):

    budget_normal      normal vector dC/dalpha to the level surface.
    project_gradient   tangent projection of alpha.grad onto the
                       budget-preserving subspace (the orthogonal
                       complement of budget_normal).
    retraction         monotonic / binary-search retraction: add a
                       scalar shift along the cost vector so the
                       iterate returns to the constraint surface.
    vector_transport   vector transport by projection of Adam's first
                       moment onto the rotated tangent plane (a first-
                       order approximation to parallel transport,
                       paper Appendix A.5).

The same three primitives have block-separable variants for MoE expert
pruning, where each layer carries its own equality constraint.
"""

from typing import Optional, Tuple

import torch


def budget_normal(alpha: torch.Tensor,
                  costs: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normal vector to the budget surface at alpha.

    dC/dalpha_ik = w_i * p_ik * (c_k - E_i[c]),  E_i[c] = sum_k p_ik c_k.
    Same shape as alpha; the tangent space is its orthogonal complement.
    """
    probs = torch.softmax(alpha, dim=-1)
    E = (probs * costs).sum(dim=-1, keepdim=True)
    n = probs * (costs - E)
    if weights is not None:
        n = n * weights.unsqueeze(-1)
    return n


def project_gradient(alpha: torch.Tensor,
                     costs: torch.Tensor,
                     weights: Optional[torch.Tensor] = None
                     ) -> Tuple[float, float, float]:
    """Tangent projection of alpha.grad onto the budget-preserving subspace.

    Subtracts the component along the budget normal in place. Returns
    (proj_coeff, raw_norm, projected_norm) for diagnostics.
    """
    with torch.no_grad():
        n = budget_normal(alpha, costs, weights)
        nf = n.flatten()
        g = alpha.grad.flatten()
        raw_norm = g.norm().item()
        coeff = (g @ nf) / (nf @ nf + 1e-12)
        alpha.grad.sub_(coeff * n)
        proj_norm = alpha.grad.norm().item()
    return coeff.item(), raw_norm, proj_norm


def retraction(alpha: torch.Tensor,
               costs: torch.Tensor,
               target: float,
               weights: Optional[torch.Tensor] = None,
               max_iter: int = 60,
               tol: float = 0.05) -> float:
    """Monotonic / binary-search retraction back to the constraint surface.

    Adds a scalar c*costs to alpha so the budget hits target. Modifies
    alpha in place. Returns the achieved budget.

    The shift is found by bisection. The bracket starts narrow and is
    doubled outward until it contains the root; bisection then halts as
    soon as the budget is within tol of the target.
    """
    def C(shift: float) -> float:
        with torch.no_grad():
            p = torch.softmax(alpha + shift * costs, dim=-1)
            row = (p * costs).sum(dim=-1)
            return ((row * weights).sum() if weights is not None
                    else row.sum()).item()

    cur = C(0.0)
    if abs(cur - target) < tol:
        return cur

    # Adaptive bracket: expand by 2x on the side that hasn't crossed yet.
    lo, hi = -1.0, 1.0
    for _ in range(40):
        if C(lo) <= target <= C(hi):
            break
        if C(hi) < target:
            hi *= 2.0
        if C(lo) > target:
            lo *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        E = C(mid)
        if abs(E - target) < tol:
            lo = hi = mid
            break
        if E > target:
            hi = mid
        else:
            lo = mid

    c = 0.5 * (lo + hi)
    with torch.no_grad():
        alpha.add_(c * costs)
    return C(0.0)


def vector_transport(optimizer,
                     alpha: torch.Tensor,
                     costs: torch.Tensor,
                     weights: Optional[torch.Tensor] = None) -> None:
    """Vector transport (by projection) of Adam's first moment.

    Re-projects exp_avg onto the current tangent plane. As alpha moves
    along the manifold the tangent plane rotates and the old moment
    acquires a small component along the new normal; removing it keeps
    Adam's adaptive capacity from leaking into the budget direction.
    The second moment is a variance estimate without a direction and is
    left untouched. This is a first-order approximation to parallel
    transport (paper Appendix A.5).
    """
    with torch.no_grad():
        n = budget_normal(alpha, costs, weights)
        nf = n.flatten()
        nsq = (nf @ nf).item() + 1e-12
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state and 'exp_avg' in state:
                    m = state['exp_avg']
                    coeff = (m.flatten() @ nf) / nsq
                    m.sub_(coeff * n)


def project_gradient_per_layer(alpha: torch.Tensor,
                               costs: torch.Tensor,
                               n_layers: int,
                               n_experts: int) -> float:
    """Block-separable tangent projection.

    Each layer carries its own equality constraint, so projection
    decomposes into independent per-layer scalar coefficients. Returns
    the mean projection coefficient across layers.
    """
    with torch.no_grad():
        n_ll = budget_normal(alpha, costs).view(n_layers, n_experts, -1)
        g_ll = alpha.grad.view(n_layers, n_experts, -1)
        dot = (g_ll * n_ll).sum(dim=(1, 2))
        nsq = (n_ll * n_ll).sum(dim=(1, 2)) + 1e-12
        coeff = dot / nsq
        alpha.grad.sub_((coeff.view(-1, 1, 1) * n_ll).reshape_as(alpha.grad))
    return coeff.mean().item()


def retraction_per_layer(alpha: torch.Tensor,
                         costs: torch.Tensor,
                         target_per_layer: float,
                         n_layers: int,
                         n_experts: int,
                         max_iter: int = 60,
                         tol: float = 1e-3) -> float:
    """Per-layer retraction: shift each layer's logits to hit its budget.

    Vectorized bisection across layers. Each layer carries its own bracket;
    brackets start at +-1 and are doubled outward on whichever side has not
    yet crossed the target. Bisection halts once every layer's budget is
    within tol of target.
    """
    with torch.no_grad():
        alpha_ll = alpha.view(n_layers, n_experts, -1)
        c = costs.view(1, 1, -1)
        target_t = torch.as_tensor(float(target_per_layer), device=alpha.device)

        def E_at(shift: torch.Tensor) -> torch.Tensor:
            shifted = alpha_ll + shift.view(-1, 1, 1) * c
            p = torch.softmax(shifted, dim=-1)
            return (p * c).sum(dim=(1, 2))

        lo = torch.full((n_layers,), -1.0, device=alpha.device)
        hi = torch.full((n_layers,),  1.0, device=alpha.device)
        for _ in range(40):
            need_lo = E_at(lo) > target_t
            need_hi = E_at(hi) < target_t
            if not need_lo.any() and not need_hi.any():
                break
            lo = torch.where(need_lo, 2.0 * lo, lo)
            hi = torch.where(need_hi, 2.0 * hi, hi)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            E = E_at(mid)
            if ((E - target_t).abs() < tol).all():
                lo = hi = mid
                break
            above = E > target_t
            hi = torch.where(above, mid, hi)
            lo = torch.where(above, lo, mid)

        final = 0.5 * (lo + hi)
        alpha_ll.add_(final.view(-1, 1, 1) * c)
        p = torch.softmax(alpha_ll, dim=-1)
        return (p * c).sum(dim=(1, 2)).mean().item()


def vector_transport_per_layer(optimizer,
                               alpha: torch.Tensor,
                               costs: torch.Tensor,
                               n_layers: int,
                               n_experts: int) -> None:
    """Block-separable vector transport of Adam's first moment."""
    with torch.no_grad():
        n_ll = budget_normal(alpha, costs).view(n_layers, n_experts, -1)
        nsq = (n_ll * n_ll).sum(dim=(1, 2)) + 1e-12
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state and 'exp_avg' in state:
                    m_ll = state['exp_avg'].view(n_layers, n_experts, -1)
                    coeff = (m_ll * n_ll).sum(dim=(1, 2)) / nsq
                    state['exp_avg'].sub_(
                        (coeff.view(-1, 1, 1) * n_ll).reshape_as(state['exp_avg']))


__all__ = [
    "budget_normal",
    "project_gradient",
    "retraction",
    "vector_transport",
    "project_gradient_per_layer",
    "retraction_per_layer",
    "vector_transport_per_layer",
]
