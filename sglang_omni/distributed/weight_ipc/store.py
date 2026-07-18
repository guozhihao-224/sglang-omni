# SPDX-License-Identifier: Apache-2.0
"""Filesystem exchange for one weight-IPC run."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

from sglang_omni.distributed.weight_ipc.types import SCHEMA_VERSION, WeightIpcBundle

BUNDLE_NAME = "bundle.pkl"
READY_NAME = "READY"
LEADER_PID_NAME = "LEADER_PID"
MANIFEST_NAME = "MANIFEST"


class WeightIpcStore:
    def __init__(self, store_dir: str | os.PathLike[str]) -> None:
        self.root = Path(store_dir)

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    @property
    def bundle_path(self) -> Path:
        return self.root / BUNDLE_NAME

    @property
    def ready_path(self) -> Path:
        return self.root / READY_NAME

    @property
    def leader_pid_path(self) -> Path:
        return self.root / LEADER_PID_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def write_bundle(self, bundle: WeightIpcBundle) -> None:
        if bundle.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={bundle.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        self.prepare()
        tmp = self.root / f"{BUNDLE_NAME}.tmp"
        payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.write_bytes(payload)
        os.replace(tmp, self.bundle_path)
        self.leader_pid_path.write_text(f"{bundle.leader_pid}\n", encoding="utf-8")
        self.manifest_path.write_text(
            (
                f"schema_version={bundle.schema_version}\n"
                f"n_tensors={len(bundle.tensors)}\n"
                f"name_digest={bundle.name_digest}\n"
                f"model_path={bundle.model_path}\n"
                f"leader_pid={bundle.leader_pid}\n"
            ),
            encoding="utf-8",
        )
        # Note (guozhihao): READY last so followers only observe a durable bundle.
        ready_tmp = self.root / f"{READY_NAME}.tmp"
        ready_tmp.write_text("1\n", encoding="utf-8")
        os.replace(ready_tmp, self.ready_path)

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready_path.is_file() and self.bundle_path.is_file():
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"weight IPC READY not found under {self.root} within {timeout_s}s"
        )

    def load_bundle(self) -> WeightIpcBundle:
        if not self.bundle_path.is_file():
            raise FileNotFoundError(f"missing weight IPC bundle: {self.bundle_path}")
        bundle = pickle.loads(self.bundle_path.read_bytes())
        if not isinstance(bundle, WeightIpcBundle):
            raise TypeError(f"unexpected bundle type: {type(bundle)!r}")
        if bundle.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: got {bundle.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        return bundle

    def read_leader_pid(self) -> int:
        return int(self.leader_pid_path.read_text(encoding="utf-8").strip())

    def cleanup(self) -> None:
        if not self.root.exists():
            return
        for name in (READY_NAME, BUNDLE_NAME, LEADER_PID_NAME, MANIFEST_NAME):
            path = self.root / name
            if path.exists():
                path.unlink()
            tmp = self.root / f"{name}.tmp"
            if tmp.exists():
                tmp.unlink()
