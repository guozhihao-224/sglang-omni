#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-process CUDA IPC weight-sharing smoke test.

Usage:
  python examples/weight_ipc/two_process_smoke.py

Leader exports a small module into a temp store; follower opens handles,
aliases parameters, and checks forward parity. Covers non-zero tensor
storage offsets via an internal view export check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from sglang_omni.distributed.weight_ipc import (
    WeightIpcStore,
    export_shared_weights,
    import_and_alias,
)
from sglang_omni.distributed.weight_ipc.cuda_handles import (
    allocation_offset_bytes,
    export_storage,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(128, 32)
        self.proj = nn.Linear(32, 16, bias=False)
        # Tied weight: share storage between embed and an alias parameter.
        self.tied = nn.Parameter(self.embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x))


def _follower(store_dir: Path, device: int) -> None:
    torch.cuda.set_device(device)
    store = WeightIpcStore(store_dir)
    store.wait_ready(timeout_s=60.0)
    bundle = store.load_bundle()

    model = TinyModel().to(device=device, dtype=torch.float32)
    # Destroy local weight contents; alias must restore leader values.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    import_and_alias(
        model,
        bundle,
        model_path="smoke://tiny",
        model_revision="v1",
    )
    x = torch.arange(8, device=device)
    y = model(x)
    payload = {
        "y_sum": float(y.sum().item()),
        "y_mean": float(y.mean().item()),
        "w0": float(model.embed.weight[0, 0].item()),
    }
    print(f"FOLLOWER_RESULT {payload['y_sum']:.6f} {payload['y_mean']:.6f} {payload['w0']:.6f}")


def _check_nonzero_offset_roundtrip(device: int) -> None:
    big = torch.empty(1024 * 1024, dtype=torch.float32, device=device)
    view = big[2048 : 2048 + 64]
    view.copy_(torch.linspace(0, 1, 64, device=device))
    offset = allocation_offset_bytes(view)
    if offset == 0:
        raise RuntimeError("expected non-zero allocation_offset_bytes for view")
    shared = export_storage(view)
    # Re-open in-process is unsupported; only validate metadata here.
    if shared.storage_size_bytes <= 0:
        raise RuntimeError("invalid shared storage size")
    print(f"OFFSET_CHECK allocation_offset_bytes={offset}")


def _leader(store_dir: Path, device: int) -> int:
    torch.cuda.set_device(device)
    _check_nonzero_offset_roundtrip(device)

    model = TinyModel().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)

    store = WeightIpcStore(store_dir)
    bundle = export_shared_weights(
        model,
        model_path="smoke://tiny",
        model_revision="v1",
    )
    store.write_bundle(bundle)

    x = torch.arange(8, device=device)
    y = model(x)
    expected = (
        float(y.sum().item()),
        float(y.mean().item()),
        float(model.embed.weight[0, 0].item()),
    )
    print(
        f"LEADER_RESULT {expected[0]:.6f} {expected[1]:.6f} {expected[2]:.6f}",
        flush=True,
    )

    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not prev else f"{repo_root}{os.pathsep}{prev}"
    follower = subprocess.run(
        [
            sys.executable,
            __file__,
            "--follower",
            "--store",
            str(store_dir),
            "--device",
            str(device),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    print(follower.stdout)
    if follower.returncode != 0:
        print(follower.stderr, file=sys.stderr)
        raise SystemExit(follower.returncode)

    line = [ln for ln in follower.stdout.splitlines() if ln.startswith("FOLLOWER_RESULT")]
    if not line:
        raise RuntimeError("follower did not report FOLLOWER_RESULT")
    parts = line[-1].split()
    got = (float(parts[1]), float(parts[2]), float(parts[3]))
    if any(abs(a - b) > 1e-5 for a, b in zip(expected, got, strict=True)):
        raise RuntimeError(f"parity mismatch leader={expected} follower={got}")
    print("SMOKE_OK")
    # Keep model alive until follower exits (subprocess already done).
    assert model is not None
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--follower", action="store_true")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2

    if args.follower:
        if args.store is None:
            raise SystemExit("--store is required for follower mode")
        _follower(args.store, args.device)
        return 0

    with tempfile.TemporaryDirectory(prefix="weight-ipc-smoke-") as tmp:
        return _leader(Path(tmp), args.device)


if __name__ == "__main__":
    raise SystemExit(main())
