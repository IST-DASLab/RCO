"""Monkey-patch vLLM to support heterogeneous per-layer expert counts.

Use:
    import vllm_pruned_patch
    vllm_pruned_patch.apply()
    # ...then any vLLM import / engine / lcb_runner call

The patch reads config.per_layer_num_experts (a list of length num_hidden_layers)
written by run_build_checkpoint.py prune --prune-mode remove with a
heterogeneous allocation. If that field is missing the patch is a no-op,
so it's safe to apply unconditionally on any checkpoint.

Currently targets Qwen3NextSparseMoeBlock (vLLM 0.16.x). EP (expert
parallelism) is NOT supported with heterogeneous counts: keep
--enable-eplb off (the default) and use TP-only.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply():
    """Idempotently install the patch. Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.model_executor.models import qwen3_next as qm
    from vllm.model_executor.models.utils import extract_layer_index

    OriginalBlock = qm.Qwen3NextSparseMoeBlock
    original_init = OriginalBlock.__init__

    def patched_init(self, vllm_config, prefix: str = ""):
        cfg = vllm_config.model_config.hf_config
        per_layer = getattr(cfg, "per_layer_num_experts", None)
        if per_layer is None:
            # No pruning metadata, fall through to original behavior.
            return original_init(self, vllm_config, prefix=prefix)

        layer_idx = extract_layer_index(prefix)
        n_local = int(per_layer[layer_idx])
        original_n = cfg.num_experts

        # Temporarily override num_experts on the (shared) hf_config so the
        # original __init__ constructs gate + FusedMoE with the right size.
        # Restore immediately after to avoid bleeding into other layers.
        try:
            cfg.num_experts = n_local
            original_init(self, vllm_config, prefix=prefix)
        finally:
            cfg.num_experts = original_n

    qm.Qwen3NextSparseMoeBlock.__init__ = patched_init

    # Patch get_expert_mapping to use max kept count (smaller mapping table).
    OriginalLM = qm.Qwen3NextForCausalLM
    original_get_expert_mapping = OriginalLM.get_expert_mapping

    def patched_get_expert_mapping(self):
        per_layer = getattr(self.config, "per_layer_num_experts", None)
        if per_layer is None:
            return original_get_expert_mapping(self)
        from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
            SharedFusedMoE,
        )
        # The saved checkpoint always has expert ids 0..max-1 across layers
        # (each layer is re-indexed densely in run_build_checkpoint.py).
        max_n = max(int(x) for x in per_layer)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=max_n,
            num_redundant_experts=self.num_redundant_experts,
        )

    qm.Qwen3NextForCausalLM.get_expert_mapping = patched_get_expert_mapping

    _PATCHED = True
    logger.info(
        "Applied Qwen3-Next pruned-model patch (heterogeneous per-layer expert counts)."
    )
