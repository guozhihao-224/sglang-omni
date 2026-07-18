# SPDX-License-Identifier: Apache-2.0
"""CUDA e2e: leader export / follower alias, including non-zero allocation offset."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

from sglang_omni.distributed.weight_ipc.cuda_handles import allocation_offset_bytes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="weight IPC CUDA tests require CUDA"
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not prev else f"{_REPO_ROOT}{os.pathsep}{prev}"
    )
    return env


def _wait_for(path: Path, proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(
                f"process exited {proc.returncode} before {path} appeared\n"
                f"stdout:\n{out}\nstderr:\n{err}"
            )
        time.sleep(0.05)
    proc.kill()
    out, err = proc.communicate()
    raise AssertionError(f"timeout waiting for {path}\nstdout:\n{out}\nstderr:\n{err}")


def test_export_import_module_alias_subprocess(tmp_path: Path) -> None:
    """One CUDA e2e: store export/import + forward parity + non-zero offset meta."""
    # Same-process offset sanity (no second subprocess path).
    big = torch.empty(256 * 1024, dtype=torch.float32, device="cuda:0")
    view = big[4096 : 4096 + 128]
    assert allocation_offset_bytes(view) != 0

    store_dir = tmp_path / "weight_ipc"
    helper = tmp_path / "helper.py"
    helper.write_text(
        textwrap.dedent(
            """
            import argparse
            import time
            import torch
            import torch.nn as nn
            from pathlib import Path
            from sglang_omni.distributed.weight_ipc import (
                WeightIpcStore,
                export_shared_weights,
                import_and_alias,
            )

            class Tiny(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(8, 4, bias=False)
                def forward(self, x):
                    return self.fc(x)

            def wait_file(path: Path, timeout_s: float = 60.0) -> None:
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    if path.exists():
                        return
                    time.sleep(0.05)
                raise SystemExit(f'timeout waiting for {path}')

            def main():
                p = argparse.ArgumentParser()
                p.add_argument('--role', choices=['leader', 'follower'])
                p.add_argument('--store', type=Path)
                args = p.parse_args()
                torch.cuda.set_device(0)
                if args.role == 'leader':
                    model = Tiny().cuda()
                    with torch.no_grad():
                        model.fc.weight.normal_()
                    store = WeightIpcStore(args.store)
                    bundle = export_shared_weights(
                        model, model_path='t', model_revision='1'
                    )
                    assert any(m.allocation_offset_bytes >= 0 for m in bundle.tensors)
                    store.write_bundle(bundle)
                    x = torch.randn(2, 8, device='cuda')
                    y = model(x)
                    torch.save(
                        {
                            'x': x.cpu(),
                            'y': y.cpu(),
                            'w': model.fc.weight.detach().cpu(),
                        },
                        args.store / 'expected.pt',
                    )
                    (args.store / 'leader_ready').write_text('1')
                    wait_file(args.store / 'follower_done')
                    print('LEADER_OK', flush=True)
                else:
                    wait_file(args.store / 'leader_ready')
                    store = WeightIpcStore(args.store)
                    store.wait_ready(30)
                    bundle = store.load_bundle()
                    model = Tiny().cuda()
                    with torch.no_grad():
                        model.fc.weight.zero_()
                    import_and_alias(
                        model, bundle, model_path='t', model_revision='1'
                    )
                    expected = torch.load(args.store / 'expected.pt', weights_only=True)
                    x = expected['x'].cuda()
                    y = model(x)
                    assert torch.allclose(y.cpu(), expected['y'], atol=0, rtol=0)
                    assert torch.allclose(
                        model.fc.weight.cpu(), expected['w'], atol=0, rtol=0
                    )
                    (args.store / 'follower_done').write_text('1')
                    print('FOLLOWER_OK', flush=True)

            if __name__ == '__main__':
                main()
            """
        ),
        encoding="utf-8",
    )

    leader = subprocess.Popen(
        [
            sys.executable,
            str(helper),
            "--role",
            "leader",
            "--store",
            str(store_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_subprocess_env(),
    )
    _wait_for(store_dir / "leader_ready", leader)

    follower = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--role",
            "follower",
            "--store",
            str(store_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(),
    )
    assert follower.returncode == 0, follower.stdout + "\n" + follower.stderr
    assert "FOLLOWER_OK" in follower.stdout

    out, err = leader.communicate(timeout=60)
    assert leader.returncode == 0, out + "\n" + err
    assert "LEADER_OK" in out
