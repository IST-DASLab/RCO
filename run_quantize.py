
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))
import os
import argparse
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_data
from quant import dist_utils
from quant.quantizer import Quantizer


def parse_args():
    parser = argparse.ArgumentParser(description="One-shot quantization with parallel GPTQ.")
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being quantized",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        "--quantizable_modules",
        type=str,
        required=True,
        help="Regex for modules to quantize",
    )
    parser.add_argument(
        "--pre_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        "--block_modules",
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    parser.add_argument(
        "--post_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules after transformer blocks",
    )
    ## Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=int(2**23), type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--bitwidth_options",
        nargs="+",
        type=int,
        required=True,
        help="List of bitwidths to quantize the model.",
    )
    parser.add_argument(
        "--calibration_bitwidth",
        type=int,
        required=True,
        help="Quantization bitwidth loaded to produce hessian. Must be in bitwidth_options.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--act_order",
        action="store_true",
        help="Whether to permute in activation order.",
    )
    parser.add_argument("--sym", action="store_true", help="Whether to use symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--block_size", type=int, default=128)
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Log to W&B")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of low_cpu_mem_usage"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--new_eval", action="store_true", help="whether to use new evaluation setup.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    # Save params
    parser.add_argument("--save_dir", type=str, required=True, help="where to save sparse model.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # init W&B logger
    if args.log_wandb and dist_utils.is_main():
        import wandb
        wandb.init(config=args)
    # Register VLM/aliased model types (e.g. qwen3_5) before the AutoConfig lookup.
    try:
        from models import _register_missing_model_types
        _register_missing_model_types()
    except Exception:
        pass
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    print(model)
    if not args.cpu_offload_modules:
        model = model.to(device)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    # Take slice (if running on multiple workers)
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(calibration_data) // world_size
        calibration_data = calibration_data[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    # Quantizer
    if args.calibration_bitwidth not in args.bitwidth_options:
        raise ValueError(f"Calibration bitwidth {args.calibration_bitwidth} is not in bitwidth_options.")
    # Move calibration_bitwidth to last position (last bitwidth is used for hessian)
    args.bitwidth_options = [bits for bits in args.bitwidth_options if bits != args.calibration_bitwidth] + [
        args.calibration_bitwidth
    ]
    dist_utils.print_on_main(f"Bitwidth options: {args.bitwidth_options}")
    # Override save dir name
    args.save_dir = os.path.join(
        args.save_dir, args.model_name_or_path.split("/")[-1], f"{args.calibration_bitwidth}bit"
    )
    quantizer = Quantizer(
        model,
        calibration_data,
        quantizable_modules=args.quantizable_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        obq_kwargs=dict(
            rel_damp=args.rel_damp,
            block_size=args.block_size,
            perchannel=args.perchannel,
            group_size=args.group_size,
            sym=args.sym,
            act_order=args.act_order,
        ),
        save_dir=args.save_dir,
        device=device,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
        verbose=args.verbose,
    )
    # Prepare save dir
    if dist_utils.is_main():
        os.makedirs(args.save_dir, exist_ok=True)

    dist.barrier()

    t1 = time.perf_counter()
    quantizer.quantize(args.bitwidth_options, args.calibration_bitwidth)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Quantization took {(t2 - t1)} s.")


if __name__ == "__main__":
    main()
