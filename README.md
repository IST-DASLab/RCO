# RCO: Riemannian Constrained Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2605.00649-b31b1b.svg)](https://arxiv.org/abs/2605.00649)

Reference implementation for [_Model Compression with Exact Budget Constraints via Riemannian Manifolds_](https://arxiv.org/abs/2605.00649) (Helcig and Alistarh, 2026).

## What RCO solves

Budget-constrained discrete assignment: pick one of K options for each of N groups so as to minimize a non-decomposable objective L (e.g. model loss after compression) subject to an exact equality budget. Because L does not factor across groups, dynamic programming alone can only optimize per-group proxies of the true objective, and penalty / Lagrangian relaxations satisfy the budget only approximately.

## The budget manifold

The discrete assignment is relaxed into softmax probabilities p_i = softmax(alpha_i) over per-group logits alpha_i in R^K. With raw group weights w_i > 0 (e.g. parameter count of layer i) and a single shared option-cost vector c in R^K (e.g. bits per parameter), the total expected cost is

```
C(alpha) = sum_i  w_i * sum_k  p_ik * c_k.
```

The constraint surface for total budget B is then

```
M = { alpha in R^(NK)  :  C(alpha) = B },
```

a smooth (NK - 1)-dimensional Riemannian submanifold of full unconstrained logit space (paper Sec. 2, Prop. 1). The per-group simplex structure is implicit in the softmax parameterization, not a separate constraint.

## The optimizer

Three manifold operations wrap a standard Adam step (paper Sec. 3, Algorithm 1):

- **Tangent projection.** Subtract the gradient's component along the closed-form budget normal (dC/dalpha)_ik = w_i * p_ik * (c_k - E_{p_i}[c]) (Prop. 2). Every Adam step is then budget-preserving by construction.
- **Retraction.** Curvature and Adam's per-coordinate scaling produce residual drift off M. RCO corrects it by shifting all logits along the cost vector, alpha' = alpha + t * c, and binary-searching the scalar t so that C(alpha') = B. The map t -> C(alpha + t*c) is monotone (Prop. 3), so the search converges to machine precision in O(log) steps.
- **Vector transport.** After retraction, Adam's first moment is re-projected onto the new tangent plane via the same inner product, so momentum stays tangent across iterations.

The discrete forward pass uses **Gumbel-STE with a budget-constrained DP solver**: Gumbel noise is added at an annealed temperature tau, an exact multiple-choice knapsack solver picks the best feasible discrete assignment in O(N*K*B), and STE gradients flow through the softmax of the same perturbed logits. Tangent projection eliminates any constraint-violating component the STE estimator introduces (Prop. 6), so the budget holds exactly throughout optimization regardless of estimator bias.

## What this repo does

It applies RCO to two LLM compression problems and ships the full pipeline (build the multi-bitwidth GPTQ database -> RCO search -> materialize a HuggingFace checkpoint):

- **Mixed-precision quantization** (paper Sec. 4.3-4.4). Groups = linear layers, options = bitwidths in {2,...,8} with c_k = k, w_i = layer parameter count, B = total-bit budget.
- **MoE expert pruning** (paper Sec. 4.2). Groups = MoE experts, options = {keep, prune} with costs (0, 1), w_i = 1, B = total number of pruned experts. Routing weights are scaled by STE survival masks during the forward pass.

The paper reports that on synthetic MCKP and on LLM compression RCO matches or exceeds evolutionary search baselines (EvoPress for quantization, EvoESAP for pruning) at 3-16x lower wall-clock, while keeping |C(alpha) - B| below floating-point noise throughout optimization.

## Setup

```bash
pip install -r requirements.txt
```

Each top-level script (run_*.py and rco_*.py) adds src/ to sys.path on the first line of code, so you can run them directly from the repo root without an editable install.

## Repository layout

```
rco-release/
  run_quantize.py             Build the multi-bitwidth GPTQ layer database.
  run_split_checkpoint.py      Split a quantized HF checkpoint into per-layer
                          tensors, populating one bitwidth slot of the
                          database (compressed-tensors source only).
  rco_search_quant.py     RCO mixed-precision quantization search.
  rco_search_prune.py     RCO MoE expert-pruning search.
  run_build_checkpoint.py Materialize a quant / prune HF checkpoint.

  scripts/                Shell wrappers with paper-default flags for each
                          of the five top-level Python entry points.
  loaders/                Optional runtime patches for serving heterogeneous
                          pruned checkpoints in vLLM (vllm_pruned_patch.py,
                          sitecustomize.py).

  src/
    manifold.py           Budget-manifold primitives: budget_normal,
                          project_gradient (tangent projection), retraction,
                          vector_transport. Block-separable variants for the
                          MoE constraint.
    search/quant.py       InterpolatedModel + projected Gumbel-STE optimizer
                          for bitwidth assignment.
    search/prune.py       MoEPruneWrapper + projected Gumbel-STE optimizer
                          for expert pruning.
    quant/                Cholesky-based OBQ kernels, block-sequential
                          quantization pipeline, qparams sidecar I/O.
    data.py               Calibration data loaders (plain-text + chat).
    metrics.py            Training-loop losses + post-hoc eval metrics
                          (compute_kl_loss, compute_ce_loss, compute_perplexity).
    grouping.py           Group construction (layer / per-expert / regex).
    store.py              On-disk per-layer database accessor.
    models.py             Model / tokenizer loading helpers.
    unfuse.py             Splits fused 3D MoE experts into per-expert
                          modules, for models that ship a fused layout.
    common.py             Memory + parameter-counting utilities and shared
                          loss helpers (compute_kl_loss,
                          compute_reference_log_probs, get_input_device).

  README.md, requirements.txt, CITATION.cff
```

### Entry points at a glance

| Script                  | Stage       | Reads                          | Writes                                            |
|-------------------------|-------------|--------------------------------|---------------------------------------------------|
| run_quantize.py             | Database    | HF model + calibration data    | <layer>/<bw>.pth, <layer>/<bw>_qparams.pt         |
| run_split_checkpoint.py      | Database    | HF compressed-tensors model    | <layer>/<bw>.pth, <layer>/<bw>_qparams.pt (one bw)|
| rco_search_quant.py     | Search      | <layer-dir>, HF model          | assignment.json (per-layer bitwidth)              |
| rco_search_prune.py     | Search      | HF model + calibration data    | mask.pt (per-layer expert keep/prune)             |
| run_build_checkpoint.py | Materialize | assignment.json or mask.pt     | HF checkpoint (bf16, packed, or pruned)           |

Each script also has a shell wrapper under scripts/ with the paper's default flags; the wrappers all forward to the Python entry point.

### Calibration datasets

data.get_data supports both plain-text and chat-formatted sources:

- Plain text: wikitext2, c4, fineweb_edu. Returns List[Tensor] with one (1, seq_length) chunk per entry.
- Chat (in data.MASKED_DATASETS): evol_codealpaca (code instructions), tulu_math (math reasoning). Returns (seqs, masks) stacked tensors where mask=1.0 marks assistant-answer tokens. Useful for calibrating code / math models with answer-only loss.

rco_search_quant.py and rco_search_prune.py both accept any of these names via --calibration-data. rco_search_prune.py also accepts '+'-joined mixtures like fineweb_edu+evol_codealpaca to combine sources at equal sample counts.

## Pipeline

### 1. Build the multi-bitwidth GPTQ database

Per layer, the pipeline writes both a dequantized fake-quant tensor (<bw>.pth, consumed by the search) and a qparams sidecar (<bw>_qparams.pt: integer codes + scales + zeros + perm + meta, consumed by checkpoint packers).

```bash
scripts/run_quantize.sh Qwen/Qwen3-8B $RCO_DATA_ROOT/qwen3_8b_db 8
```

Or directly:

```bash
torchrun --nproc-per-node=8 run_quantize.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --quantizable_modules '.*(q|k|v|o|gate|up|down)_proj$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    --post_block_modules model.norm lm_head \
    --calibration_data fineweb_edu \
    --bitwidth_options 2 3 4 5 6 7 8 \
    --calibration_bitwidth 4 \
    --group_size 128 \
    --save_dir $RCO_DATA_ROOT/qwen3_8b_db
```

**Seeding a bitwidth from an existing HF checkpoint.** If a pre-quantized checkpoint already exists for one of the bitwidths you care about (e.g. a uniform W4 model on the Hub), run_split_checkpoint.py reads it via the compressed-tensors library, decompresses each linear layer to bf16, and writes both files into the database: <layer>/<bw>.pth (dequantized fake-quant tensor) and <layer>/<bw>_qparams.pt (sidecar with integer codes / scales / zeros / meta). The integer codes are extracted from the packed buffers *before* decompression and converted to our schema, so the database produced by import is interchangeable with one produced by run_quantize.py. This populates only one bitwidth slot; other bitwidths still come from run_quantize.py.

```bash
scripts/run_split_checkpoint.sh <hf_model_id> $RCO_DATA_ROOT/qwen3_8b_db 4
```

### 2a. Mixed-precision quantization search

```bash
scripts/run_search_quant.sh Qwen/Qwen3-8B $RCO_DATA_ROOT/qwen3_8b_db 2.5
```

Materialize the checkpoint:

```bash
scripts/run_build_checkpoint.sh quant \
    Qwen/Qwen3-8B \
    $RCO_DATA_ROOT/qwen3_8b_db/<assignment>.json \
    $RCO_DATA_ROOT/qwen3_8b_db \
    ./qwen3_8b_2p5bit
```

run_build_checkpoint.py quant has two output formats, switched via --format:

- --format fake-quant (default): each linear's .weight is replaced with the dequantized fake-quant tensor for the chosen bitwidth, and a format: "dense" / quantization_status: "frozen" quantization_config block is written into config.json. Output is plain bf16 safetensors plus metadata; loadable by any HF tool, no size savings on disk.
- --format compressed-tensors: each linear's packed buffers (weight_packed / weight_scale / weight_zero_point / weight_shape) are repacked from the database's <bw>_qparams.pt sidecars, and the quantization_config is written as format: "pack-quantized" / quantization_status: "compressed". Output is a real packed compressed-tensors checkpoint with on-disk size matching the average bitwidth, loadable directly by vLLM / compressed-tensors-aware loaders.

### 2b. MoE expert pruning

```bash
scripts/run_search_prune.sh allenai/OLMoE-1B-7B-0125-Instruct 0.25 mask.pt 300
scripts/run_build_checkpoint.sh prune \
    allenai/OLMoE-1B-7B-0125-Instruct \
    mask.pt \
    ./olmoe_25pct_pruned
```

run_build_checkpoint.py prune has two output modes, switched via --prune-mode:

- --prune-mode zero (default): pruned experts and their router rows are zeroed in place. Tensor shapes and on-disk size are unchanged. Loads in any stock HF / vLLM with no patches.
- --prune-mode remove: pruned experts are physically dropped from mlp.experts and the router is shrunk to match. If every layer keeps the same count the model's num_experts is rewritten and the result loads in stock HF / vLLM. Otherwise a per_layer_num_experts list is written into config.json and the checkpoint needs a per-layer loader patch to be served.

For heterogeneous pruned checkpoints, the loaders/ directory ships vllm_pruned_patch.py (targets Qwen3-Next) which teaches vLLM to build each MoE layer with its own kept-count by reading per_layer_num_experts. Apply manually:

```python
import vllm_pruned_patch
vllm_pruned_patch.apply()
# ...then any vLLM import / engine call
```

or auto-apply by pointing PYTHONPATH at loaders/ (sitecustomize.py picks it up):

```bash
PYTHONPATH=loaders vllm serve <path-to-pruned-checkpoint>
```

## qparams sidecar format

Each <layer>/<bw>_qparams.pt is a torch-pickled dict:

| key          | dtype / shape                              | meaning                                |
|--------------|--------------------------------------------|----------------------------------------|
| qweight    | uint8 [d_row, d_col]                     | integer codes, [0, 2**bits - 1]      |
| scales     | original dtype [d_row, n_groups]         | per-(row, group) scale                 |
| zeros      | original dtype [d_row, n_groups]         | per-(row, group) zero-point            |
| perm       | int64 [d_col] or None                  | act-order permutation, if any          |
| bits       | int                                        | 2..8                                   |
| group_size | int                                        | along input dim                        |
| sym        | bool                                       | symmetric grid?                        |
| act_order  | bool                                       | activation-order GPTQ used?            |
| shape      | tuple                                      | original (out_features, in_features) |
| dtype      | str                                        | original weight dtype (e.g. bfloat16)|
| schema     | int                                        | format version                         |

Reconstruction (column-permuted form): W_q = scales[:, group_idx] * (qweight - zeros[:, group_idx]), where group_idx[c] = c // group_size. See gptq.dequantize_from_qparams.

## Acknowledgements

The Cholesky-based OBQ quantization kernels in src/quant/ derive from GPTQ ([Frantar et al., 2023](https://arxiv.org/abs/2210.17323)).

## Citation

```bibtex
@misc{helcig2026rco,
  title         = {Model Compression with Exact Budget Constraints via Riemannian Manifolds},
  author        = {Michael Helcig and Dan Alistarh},
  year          = {2026},
  eprint        = {2605.00649},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
}
```

