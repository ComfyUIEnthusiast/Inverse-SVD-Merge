#!/usr/bin/env python3
"""
GPU-accelerated LoRA merger.

Merge two LoRAs into a single LoRA:

    merged = alpha * LoRA1 - beta * LoRA2

The script:
    - Loads LoRAs on CPU.
    - Processes one layer at a time.
    - Moves the current layer to GPU.
    - Performs matrix multiplication and SVD on GPU.
    - Moves the resulting LoRA tensors back to CPU.
    - Saves the final LoRA as safetensors.

This minimizes VRAM usage while accelerating the expensive operations.

Example:

    python InverseMerge.py lora1.safetensors lora2.safetensors merged.safetensors

Custom strengths:

    python InverseMerge.py \
        lora1.safetensors \
        lora2.safetensors \
        merged.safetensors \
        --alpha 1.0 \
        --beta 1.0

Specify GPU:

    python InverseMerge.py \
        lora1.safetensors \
        lora2.safetensors \
        merged.safetensors \
        --device cuda:0

CPU fallback:

    python InverseMerge.py \
        lora1.safetensors \
        lora2.safetensors \
        merged.safetensors \
        --device cpu

Requirements:

    pip install torch safetensors
"""

import argparse
import gc
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def get_lora_pairs(state_dict):
    """
    Find LoRA A/B tensor pairs.

    Supports:

        lora_unet_xxx.lora_A.weight
        lora_unet_xxx.lora_B.weight

    and Kohya-style:

        lora_unet_xxx.lora_down.weight
        lora_unet_xxx.lora_up.weight
    """

    pairs = {}

    for key in state_dict:
        if ".lora_A.weight" in key:
            prefix = key.replace(".lora_A.weight", "")
            b_key = prefix + ".lora_B.weight"

            if b_key in state_dict:
                pairs[prefix] = (key, b_key)

        elif ".lora_down.weight" in key:
            prefix = key.replace(".lora_down.weight", "")
            b_key = prefix + ".lora_up.weight"

            if b_key in state_dict:
                pairs[prefix] = (key, b_key)

    return pairs


def print_gpu_info(device):
    """Print information about the selected CUDA device."""

    if device.type != "cuda":
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False.\n"
            "Make sure you installed a CUDA-enabled PyTorch build."
        )

    index = device.index
    if index is None:
        index = torch.cuda.current_device()

    name = torch.cuda.get_device_name(index)

    total = torch.cuda.get_device_properties(index).total_memory
    total_gb = total / (1024 ** 3)

    print(f"GPU      : {name}")
    print(f"VRAM     : {total_gb:.2f} GB")
    print(f"CUDA     : {torch.version.cuda}")
    print()


def gpu_memory(device):
    """Return currently allocated/reserved GPU memory in GB."""

    if device.type != "cuda":
        return None

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    return (
        allocated / (1024 ** 3),
        reserved / (1024 ** 3),
    )


def cleanup_gpu(device):
    """Release temporary GPU memory."""

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()


def merge_lora_tensors(
    lora1,
    lora2,
    alpha=1.0,
    beta=1.0,
    device=torch.device("cuda"),
    output_dtype=None,
):
    """
    Merge two LoRAs:

        result = alpha * LoRA1 - beta * LoRA2

    The actual LoRA update is:

        delta_W = B @ A

    We reconstruct the full update, combine the two updates, and
    factorize the result back into LoRA form using SVD.

    Only the current layer is placed on the GPU at any given time.
    """

    pairs1 = get_lora_pairs(lora1)
    pairs2 = get_lora_pairs(lora2)

    common = sorted(set(pairs1) & set(pairs2))

    if not common:
        raise RuntimeError(
            "No matching LoRA layers were found between the two files."
        )

    print(f"Found {len(common)} matching LoRA layers.")
    print(f"Processing device: {device}")
    print()

    output = {}

    for i, prefix in enumerate(common, 1):

        a1_key, b1_key = pairs1[prefix]
        a2_key, b2_key = pairs2[prefix]

        print(f"[{i:4d}/{len(common)}] {prefix}")

        # ---------------------------------------------------------
        # Load current layer onto GPU.
        #
        # The source LoRAs remain on CPU.
        # ---------------------------------------------------------

        A1 = lora1[a1_key].float().to(device)
        B1 = lora1[b1_key].float().to(device)

        A2 = lora2[a2_key].float().to(device)
        B2 = lora2[b2_key].float().to(device)

        # ---------------------------------------------------------
        # Determine rank from original LoRA.
        # ---------------------------------------------------------

        rank1 = A1.shape[0]
        rank2 = A2.shape[0]

        # Preserve the larger original rank.
        requested_rank = max(rank1, rank2)

        # ---------------------------------------------------------
        # Reconstruct LoRA updates.
        #
        # Standard linear LoRA:
        #
        #     delta_W = B @ A
        # ---------------------------------------------------------

        if A1.ndim == 2:

            delta1 = B1 @ A1
            delta2 = B2 @ A2

            delta1_2d = delta1.reshape(delta1.shape[0], -1)
            delta2_2d = delta2.reshape(delta2.shape[0], -1)

        else:

            # -----------------------------------------------------
            # Convolutional LoRA.
            #
            # Flatten input dimensions so that:
            #
            #     B @ A
            #
            # can be performed as a matrix multiplication.
            # -----------------------------------------------------

            B1_2d = B1.reshape(B1.shape[0], -1)
            B2_2d = B2.reshape(B2.shape[0], -1)

            A1_2d = A1.reshape(A1.shape[0], -1)
            A2_2d = A2.reshape(A2.shape[0], -1)

            delta1_2d = B1_2d @ A1_2d
            delta2_2d = B2_2d @ A2_2d

        # ---------------------------------------------------------
        # Verify dimensions.
        # ---------------------------------------------------------

        if delta1_2d.shape != delta2_2d.shape:
            raise ValueError(
                f"Shape mismatch in layer '{prefix}': "
                f"{delta1_2d.shape} vs {delta2_2d.shape}"
            )

        # ---------------------------------------------------------
        # Combine the two LoRA directions.
        #
        #     merged = alpha * LoRA1 - beta * LoRA2
        # ---------------------------------------------------------

        merged = (
            alpha * delta1_2d
            - beta * delta2_2d
        )

        # ---------------------------------------------------------
        # Free the individual reconstructed deltas.
        # They are no longer needed.
        # ---------------------------------------------------------

        del delta1
        del delta2

        # ---------------------------------------------------------
        # SVD ON GPU.
        #
        # merged = U @ diag(S) @ Vh
        # ---------------------------------------------------------

        U, S, Vh = torch.linalg.svd(
            merged,
            full_matrices=False,
        )

        # ---------------------------------------------------------
        # Determine final rank.
        #
        # Can't exceed the available SVD rank.
        # ---------------------------------------------------------

        rank = min(
            requested_rank,
            S.shape[0],
            S.shape[1] if S.ndim > 1 else S.shape[0],
        )

        # For S, shape is [min(m, n)], so this is sufficient.
        rank = min(requested_rank, S.shape[0])

        U = U[:, :rank].contiguous()
        S = S[:rank].contiguous()
        Vh = Vh[:rank, :].contiguous()

        B = (U * S.unsqueeze(0)).contiguous()
        A = Vh.contiguous()

        # ---------------------------------------------------------
        # Move output back to CPU.
        #
        # Keep the same dtype as LoRA1 unless --dtype was supplied.
        # ---------------------------------------------------------

        if output_dtype is None:
            a_dtype = lora1[a1_key].dtype
            b_dtype = lora1[b1_key].dtype
        else:
            a_dtype = output_dtype
            b_dtype = output_dtype

        A_cpu = A.to(
            device="cpu",
            dtype=a_dtype,
        ).contiguous()

        B_cpu = B.to(
            device="cpu",
            dtype=b_dtype,
        ).contiguous()

        # Preserve LoRA1's naming convention.

        output[a1_key] = A_cpu.contiguous()
        output[b1_key] = B_cpu.contiguous()

        # ---------------------------------------------------------
        # Free GPU tensors for this layer.
        # ---------------------------------------------------------

        del A1
        del B1
        del A2
        del B2

        del merged
        del U
        del S
        del Vh
        del A
        del B

        cleanup_gpu(device)

        # ---------------------------------------------------------
        # Display memory usage.
        # ---------------------------------------------------------

        mem = gpu_memory(device)

        if mem is not None:
            allocated, reserved = mem
            print(
                f"       rank={rank} | "
                f"GPU allocated={allocated:.2f} GB | "
                f"reserved={reserved:.2f} GB"
            )
        else:
            print(f"       rank={rank}")

    return output


def main():

    parser = argparse.ArgumentParser(
        description=(
            "GPU-accelerated additive LoRA merger: "
            "alpha*LoRA1 - beta*LoRA2"
        )
    )

    parser.add_argument(
        "lora1",
        type=Path,
        help="Positive LoRA",
    )

    parser.add_argument(
        "lora2",
        type=Path,
        help="Negative LoRA",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output merged LoRA",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Strength of LoRA1 (default: 1.0)",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Strength of LoRA2 (default: 1.0)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "Processing device. "
            "Examples: cuda, cuda:0, cuda:1, cpu. "
            "Default: cuda"
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=[
            "original",
            "float16",
            "bfloat16",
            "float32",
        ],
        default="original",
        help=(
            "Output dtype. "
            "Default: original dtype from LoRA1."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Validate input files.
    # -------------------------------------------------------------

    if not args.lora1.exists():
        raise FileNotFoundError(
            f"LoRA1 not found: {args.lora1}"
        )

    if not args.lora2.exists():
        raise FileNotFoundError(
            f"LoRA2 not found: {args.lora2}"
        )

    # -------------------------------------------------------------
    # Select device.
    # -------------------------------------------------------------

    if args.device.startswith("cuda"):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available.\n\n"
                "Check your PyTorch installation with:\n"
                "    python -c "
                "\"import torch; "
                "print(torch.cuda.is_available()); "
                "print(torch.version.cuda)\""
            )

        device = torch.device(args.device)

    else:
        device = torch.device(args.device)

    # -------------------------------------------------------------
    # Determine output dtype.
    # -------------------------------------------------------------

    if args.dtype == "original":
        output_dtype = None

    elif args.dtype == "float16":
        output_dtype = torch.float16

    elif args.dtype == "bfloat16":
        output_dtype = torch.bfloat16

    elif args.dtype == "float32":
        output_dtype = torch.float32

    else:
        raise ValueError(
            f"Unsupported dtype: {args.dtype}"
        )

    # -------------------------------------------------------------
    # Header.
    # -------------------------------------------------------------

    print("=" * 70)
    print("GPU-Accelerated LoRA Additive Merger")
    print("=" * 70)

    print(f"LoRA 1  : {args.lora1}")
    print(f"LoRA 2  : {args.lora2}")
    print(f"Alpha   : {args.alpha}")
    print(f"Beta    : {args.beta}")
    print(f"Output  : {args.output}")
    print(f"Device  : {device}")
    print(f"Dtype   : {args.dtype}")

    print()
    print("Formula:")
    print(
        f"    merged = "
        f"{args.alpha} * LoRA1 - "
        f"{args.beta} * LoRA2"
    )

    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # GPU information.
    # -------------------------------------------------------------

    print_gpu_info(device)

    # -------------------------------------------------------------
    # Load LoRAs.
    #
    # IMPORTANT:
    # They stay on CPU.
    #
    # Individual layers are transferred to GPU during processing.
    # -------------------------------------------------------------

    print("Loading LoRA 1 on CPU...")

    lora1 = load_file(
        str(args.lora1),
        device="cpu",
    )

    print(
        f"Loaded {len(lora1)} tensors."
    )

    print()

    print("Loading LoRA 2 on CPU...")

    lora2 = load_file(
        str(args.lora2),
        device="cpu",
    )

    print(
        f"Loaded {len(lora2)} tensors."
    )

    print()

    # -------------------------------------------------------------
    # Merge.
    # -------------------------------------------------------------

    print("Merging...")
    print()

    merged = merge_lora_tensors(
        lora1,
        lora2,
        alpha=args.alpha,
        beta=args.beta,
        device=device,
        output_dtype=output_dtype,
    )

    # -------------------------------------------------------------
    # Metadata.
    # -------------------------------------------------------------

    metadata = {
        "ss_merge_algorithm": "additive_lora_svd_gpu",
        "ss_merge_lora1": args.lora1.name,
        "ss_merge_lora2": args.lora2.name,
        "ss_merge_alpha": str(args.alpha),
        "ss_merge_beta": str(args.beta),
        "ss_merge_device": str(device),
        "ss_merge_output_dtype": args.dtype,
    }

    # -------------------------------------------------------------
    # Create output directory.
    # -------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    print()
    print("Saving...")

    save_file(
        merged,
        str(args.output),
        metadata=metadata,
    )

    # -------------------------------------------------------------
    # Cleanup.
    # -------------------------------------------------------------

    del merged
    del lora1
    del lora2

    cleanup_gpu(device)

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()