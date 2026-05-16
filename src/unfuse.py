"""
Unfuse packed MoE expert weights into individual nn.Linear modules.

Qwen3-MoE (and similar models) store expert weights as 3D parameters
(gate_up_proj: [num_experts, 2*intermediate, hidden], down_proj:
[num_experts, hidden, intermediate]) inside a single module. The GPTQ
pipeline requires individual nn.Linear layers for per-layer Hessian
collection and quantization.

This module replaces each fused expert block with an equivalent module
containing individual nn.Linear layers per expert, producing module names
like:
    model.layers.0.mlp.experts.expert_linears.0.gate_proj
    model.layers.0.mlp.experts.expert_linears.0.up_proj
    model.layers.0.mlp.experts.expert_linears.0.down_proj
    ...
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SingleExpert(nn.Module):
    """A single MoE expert with individual gate_proj, up_proj, down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int, act_fn):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnfusedExperts(nn.ModuleList):
    """
    Drop-in replacement for fused expert modules (e.g. Qwen3MoeExperts).

    Subclasses nn.ModuleList so that experts are accessible directly as
    experts[i] (and named modules report experts.0.gate_proj, etc.,
    matching the per-layer paths used by the GPTQ output and by older
    transformers versions that stored experts as nn.ModuleList).
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, act_fn):
        super().__init__([
            SingleExpert(hidden_size, intermediate_size, act_fn)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        # store act_fn outside nn.Module's child-registration path; otherwise
        # __setattr__ would pollute _modules and break len(self) / integer
        # indexing. Each SingleExpert holds its own act_fn for forward.
        object.__setattr__(self, "act_fn", act_fn)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            expert = self[expert_idx]
            current_hidden_states = expert(current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def unfuse_moe_experts(model: nn.Module) -> List[str]:
    """
    Replace all fused MoE expert modules with unfused versions in-place.

    Looks for modules with 3D gate_up_proj and down_proj parameters
    (the pattern used by Qwen3MoeExperts and similar architectures).

    Memory-efficient: copies expert weights one at a time and uses
    views where possible to avoid peak memory doubling.

    Returns a list of module paths that were replaced.
    """
    import gc

    replaced = []

    for name, module in list(model.named_modules()):
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)

        if not (
            isinstance(gate_up, (nn.Parameter, torch.Tensor))
            and isinstance(down, (nn.Parameter, torch.Tensor))
            and gate_up.dim() == 3
            and down.dim() == 3
        ):
            continue

        num_experts = gate_up.shape[0]
        fused_intermediate = gate_up.shape[1]
        hidden_size = gate_up.shape[2]
        intermediate_size = fused_intermediate // 2

        assert down.shape == (num_experts, hidden_size, intermediate_size), (
            f"Unexpected down_proj shape {down.shape}, expected "
            f"({num_experts}, {hidden_size}, {intermediate_size})"
        )

        act_fn = getattr(module, "act_fn", nn.SiLU())

        # use ModuleList __init__ so children are stored at integer keys
        # (paths like experts.0.gate_proj, not experts.expert_linears.0...).
        expert_list = []
        for i in range(num_experts):
            expert = SingleExpert.__new__(SingleExpert)
            nn.Module.__init__(expert)
            expert.act_fn = act_fn

            expert.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            expert.gate_proj.weight = nn.Parameter(gate_up.data[i, :intermediate_size, :].clone())

            expert.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            expert.up_proj.weight = nn.Parameter(gate_up.data[i, intermediate_size:, :].clone())

            expert.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            expert.down_proj.weight = nn.Parameter(down.data[i].clone())

            expert_list.append(expert)

        unfused = UnfusedExperts.__new__(UnfusedExperts)
        nn.ModuleList.__init__(unfused, expert_list)
        unfused.num_experts = num_experts
        unfused.hidden_dim = hidden_size
        unfused.intermediate_dim = intermediate_size
        unfused.act_fn = act_fn

        # free fused params before swapping the module so peak memory does
        # not double while the next layer is being processed.
        del module.gate_up_proj
        del module.down_proj
        gc.collect()

        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, name, unfused)
        else:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], unfused)

        # re-attach act_fn outside the child-registration path so it does
        # not pollute _modules and inflate len(experts) past num_experts.
        try:
            del unfused._modules["act_fn"]
        except KeyError:
            pass
        object.__setattr__(unfused, "act_fn", act_fn)

        replaced.append(name)
        logger.info(
            f"Unfused {name}: {num_experts} experts, "
            f"hidden={hidden_size}, intermediate={intermediate_size}"
        )

    gc.collect()
    return replaced
