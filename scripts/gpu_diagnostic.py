"""Phase 11 Step 1: GPU Diagnostic Report.

Measures VRAM capacity, bandwidth, and DataLoader throughput
to identify bottlenecks before applying GPU utilization fixes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("=" * 60)
    print("GPU DIAGNOSTIC REPORT")
    print("=" * 60)

    # Basic availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA NOT AVAILABLE. Cannot proceed.")

    print(f"Device name: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # cuDNN state
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")

    # Bandwidth test — saturate GPU with large matrix multiply
    device = torch.device("cuda")
    sizes = [512, 1024, 2048, 4096]
    print("\nGPU Bandwidth Test:")
    for s in sizes:
        a = torch.randn(s, s, device=device, dtype=torch.float16)
        b = torch.randn(s, s, device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            c = a @ b
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"  Size {s}x{s}: {elapsed:.3f}s, VRAM used: {mem:.3f} GB")
        del a, b, c
        torch.cuda.empty_cache()

    # DataLoader bottleneck test
    from torch.utils.data import DataLoader, TensorDataset

    dummy_X = torch.randn(50000, 60, 10)
    dummy_y = torch.randint(0, 2, (50000,))
    dataset = TensorDataset(dummy_X, dummy_y)

    print("\nDataLoader Throughput Test:")
    for nw in [0, 2, 4]:
        loader = DataLoader(
            dataset,
            batch_size=1024,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
        )
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            if i >= 20:
                break
        elapsed = time.time() - t0
        print(f"  num_workers={nw}: {elapsed:.3f}s for 20 batches")

    # Simulate fusion training VRAM usage at different batch sizes
    print("\nFusion Training VRAM Simulation:")
    for bs in [1024, 2048, 4096, 8192]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            # Simulate embeddings on GPU
            price = torch.randn(bs, 256, device=device)
            gat = torch.randn(bs, 256, device=device)
            doc = torch.randn(bs, 768, device=device)
            macro = torch.randn(bs, 32, device=device)
            surprise = torch.randn(bs, 5, device=device)
            labels = torch.randint(0, 2, (bs,), device=device)
            vol = torch.randn(bs, device=device)
            alloc = torch.cuda.memory_allocated(0) / 1e9
            peak = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"  batch_size={bs}: {alloc:.3f} GB allocated, {peak:.3f} GB peak")
            del price, gat, doc, macro, surprise, labels, vol
        except RuntimeError as e:
            print(f"  batch_size={bs}: OOM - {e}")
        torch.cuda.empty_cache()

    print("\nDIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
