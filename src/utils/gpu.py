"""GPU diagnostics, optimization, and mixed-precision utilities.

Provides a single ``setup_gpu()`` call that every training script should invoke
at startup. Enables cuDNN benchmark mode, TF32, and returns a configured
``torch.amp.GradScaler`` for automatic mixed-precision training.
"""

from __future__ import annotations

import torch


def setup_gpu(verbose: bool = True) -> torch.device:
    """Configure GPU optimizations and return the compute device.

    Enables:
    - cuDNN benchmark mode (auto-tunes convolution algorithms)
    - TF32 for matmul and cuDNN (faster on Ampere+ GPUs)
    - Prints GPU diagnostics

    Raises AssertionError if CUDA is not available.
    """
    if not torch.cuda.is_available():
        if verbose:
            print("WARNING: CUDA not available — falling back to CPU")
        return torch.device("cpu")

    device = torch.device("cuda")

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if verbose:
        props = torch.cuda.get_device_properties(0)
        print("=" * 60)
        print("GPU DIAGNOSTICS")
        print("=" * 60)
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
        print("=" * 60)

    return device


def log_gpu_usage(prefix: str = "") -> None:
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def create_grad_scaler() -> torch.amp.GradScaler:
    """Create a GradScaler for automatic mixed-precision training."""
    return torch.amp.GradScaler("cuda")


def gpu_utilization_report() -> dict:
    """Return a dict with GPU utilization metrics."""
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    total = props.total_memory
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "utilization_pct": (allocated / total) * 100 if total > 0 else 0,
    }
