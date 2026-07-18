# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for weight IPC store / digest / policy."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sglang_omni.distributed.weight_ipc.export import compute_name_digest
from sglang_omni.distributed.weight_ipc.import_ import (
    WeightIpcImportError,
    validate_bundle,
)
from sglang_omni.distributed.weight_ipc.lifecycle import pid_is_alive
from sglang_omni.distributed.weight_ipc.select import ArParametersPolicy
from sglang_omni.distributed.weight_ipc.store import WeightIpcStore
from sglang_omni.distributed.weight_ipc.types import (
    SCHEMA_VERSION,
    IpcTensorMeta,
    WeightIpcBundle,
)


def _meta(name: str, shape=(2, 2)) -> IpcTensorMeta:
    return IpcTensorMeta(
        name=name,
        shape=tuple(shape),
        stride=(shape[1], 1) if len(shape) == 2 else (1,),
        dtype="torch.float32",
        nbytes=8,
        device_index=0,
        handle=b"\x00" * 64,
        storage_size_bytes=64,
        storage_offset_bytes=0,
        allocation_offset_bytes=0,
        tensor_offset=0,
        requires_grad=False,
        ref_counter_handle=b"",
        ref_counter_offset=0,
        event_handle=b"",
        event_sync_required=False,
    )


def test_store_write_and_load_roundtrip(tmp_path: Path) -> None:
    store = WeightIpcStore(tmp_path / "weight_ipc")
    tensors = [_meta("weight"), _meta("bias", shape=(2,))]
    bundle = WeightIpcBundle(
        schema_version=SCHEMA_VERSION,
        model_path="m",
        model_revision="r",
        cuda_driver=None,
        created_unix_s=1.0,
        leader_pid=1234,
        tensors=tensors,
        name_digest=compute_name_digest(tensors),
    )
    store.write_bundle(bundle)
    assert store.ready_path.is_file()
    assert store.manifest_path.is_file()
    loaded = store.load_bundle()
    assert loaded.name_digest == bundle.name_digest
    assert loaded.leader_pid == 1234
    assert store.read_leader_pid() == 1234


def test_wait_ready_timeout(tmp_path: Path) -> None:
    store = WeightIpcStore(tmp_path / "empty")
    store.prepare()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        store.wait_ready(timeout_s=0.2)
    assert time.monotonic() - start >= 0.2


def test_validate_bundle_digest_mismatch() -> None:
    tensors = [_meta("weight")]
    bundle = WeightIpcBundle(
        schema_version=SCHEMA_VERSION,
        model_path="m",
        model_revision=None,
        cuda_driver=None,
        created_unix_s=1.0,
        leader_pid=1,
        tensors=tensors,
        name_digest="deadbeef",
    )
    with pytest.raises(WeightIpcImportError, match="name_digest"):
        validate_bundle(bundle, model_path="m", model_revision=None)


def test_validate_bundle_model_path_mismatch() -> None:
    tensors = [_meta("weight")]
    bundle = WeightIpcBundle(
        schema_version=SCHEMA_VERSION,
        model_path="a",
        model_revision=None,
        cuda_driver=None,
        created_unix_s=1.0,
        leader_pid=1,
        tensors=tensors,
        name_digest=compute_name_digest(tensors),
    )
    with pytest.raises(WeightIpcImportError, match="model_path"):
        validate_bundle(bundle, model_path="b", model_revision=None)


def test_ar_parameters_policy_skips_buffers_by_default() -> None:
    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 2))
            self.register_buffer("buf", torch.ones(2))

    # CPU tensors are skipped (CUDA-only share).
    selected = ArParametersPolicy().select(M())
    assert selected == []


def test_pid_is_alive_self() -> None:
    import os

    assert pid_is_alive(os.getpid())
    assert not pid_is_alive(2**30)
