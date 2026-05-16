"""Layer grouping for constraining layers to share a bitwidth."""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LayerGroup:
    """A set of layers constrained to share the same bitwidth."""
    group_id: int
    layer_names: List[str]
    pattern: str


def parse_group_spec(group_spec: str) -> List[List[str]]:
    """
    Parse a custom group specification string.
    
    Format: Space-separated groups, each group is comma-separated patterns.
    Example: 'mlp.gate_proj,mlp.down_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj'
    
    Returns list of groups, each group is a list of patterns.
    """
    if not group_spec or not group_spec.strip():
        return []
    
    groups = []
    for group_str in group_spec.strip().split():
        patterns = [p.strip() for p in group_str.split(',') if p.strip()]
        if patterns:
            groups.append(patterns)
    
    return groups


def build_layer_groups(
    layer_names: List[str], 
    group_patterns: List[List[str]]
) -> List[LayerGroup]:
    """
    Build layer groups based on patterns.
    
    For each transformer block, layers matching patterns in the same group
    are constrained to share the same bitwidth.
    
    Args:
        layer_names: List of all layer names (e.g., 'model.layers.0.mlp.gate_proj')
        group_patterns: List of pattern groups from parse_group_spec
    
    Returns:
        List of LayerGroup objects
    """
    if not group_patterns:
        return [LayerGroup(i, [name], name) for i, name in enumerate(layer_names)]

    block_pattern = re.compile(r'model\.layers\.(\d+)\.')

    block_groups: Dict[Tuple[int, int], List[str]] = {}
    ungrouped_layers: List[str] = []

    for layer_name in layer_names:
        match = block_pattern.search(layer_name)
        if not match:
            ungrouped_layers.append(layer_name)
            continue

        block_idx = int(match.group(1))
        layer_suffix = layer_name[match.end():]

        assigned = False
        for group_idx, patterns in enumerate(group_patterns):
            for pattern in patterns:
                if pattern in layer_suffix:
                    key = (block_idx, group_idx)
                    if key not in block_groups:
                        block_groups[key] = []
                    block_groups[key].append(layer_name)
                    assigned = True
                    break
            if assigned:
                break
        
        if not assigned:
            ungrouped_layers.append(layer_name)
    
    groups = []
    group_id = 0

    for (block_idx, pattern_idx), layers in sorted(block_groups.items()):
        pattern_str = ','.join(group_patterns[pattern_idx])
        groups.append(LayerGroup(
            group_id=group_id,
            layer_names=sorted(layers),
            pattern=f"block_{block_idx}:{pattern_str}"
        ))
        group_id += 1

    for layer_name in ungrouped_layers:
        groups.append(LayerGroup(
            group_id=group_id,
            layer_names=[layer_name],
            pattern=layer_name
        ))
        group_id += 1
    
    return groups


def build_moe_per_expert_groups(
    layer_names: List[str],
    extra_group_patterns: List[List[str]] = None,
) -> List[LayerGroup]:
    """
    MoE-aware grouping. Each expert (gate_proj + up_proj + down_proj across
    one (layer, expert_idx)) is one group. Non-expert layers fall through to
    the substring matcher in build_layer_groups using extra_group_patterns.

    Args:
        layer_names: All quantized layer names from the weight store.
        extra_group_patterns: Group spec applied only to non-expert layers
            (e.g., to share bitwidth across self_attn KQV).

    Returns:
        List of LayerGroup with expert groups first, then non-expert groups.
    """
    expert_pattern = re.compile(r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.')

    expert_layers: Dict[Tuple[int, int], List[str]] = {}
    non_expert_layers: List[str] = []
    for name in layer_names:
        m = expert_pattern.search(name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            expert_layers.setdefault(key, []).append(name)
        else:
            non_expert_layers.append(name)

    groups: List[LayerGroup] = []
    gid = 0
    for (layer_idx, expert_idx), names in sorted(expert_layers.items()):
        groups.append(LayerGroup(
            group_id=gid,
            layer_names=sorted(names),
            pattern=f"layer_{layer_idx}_expert_{expert_idx}",
        ))
        gid += 1

    non_expert_groups = build_layer_groups(
        non_expert_layers, extra_group_patterns or []
    )
    for g in non_expert_groups:
        groups.append(LayerGroup(
            group_id=gid,
            layer_names=g.layer_names,
            pattern=g.pattern,
        ))
        gid += 1

    return groups


def get_layer_to_group_mapping(groups: List[LayerGroup]) -> Dict[str, int]:
    """Create mapping from layer name to group ID."""
    mapping = {}
    for group in groups:
        for layer_name in group.layer_names:
            mapping[layer_name] = group.group_id
    return mapping


def get_group_param_count(
    group: LayerGroup, 
    param_counts: Dict[str, int]
) -> int:
    """Get total parameter count for a layer group."""
    return sum(param_counts.get(name, 1) for name in group.layer_names)
