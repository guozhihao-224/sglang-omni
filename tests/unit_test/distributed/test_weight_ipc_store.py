# SPDX-License-Identifier: Apache-2.0
"""CPU tests for weight-IPC store protocol and digest validation."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sglang_omni.distributed.weight_ipc.export import compute_name_digest
from sglang_omni.distributed.weight_ipc.import_ import (
    WeightIpcImportError,
    validate_bundle,
)
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
    loaded = store.load_bundle()
    assert loaded.name_digest == bundle.name_digest
    assert store.read_leader_pid() == 1234


def test_wait_ready_timeout(tmp_path: Path) -> None:
    store = WeightIpcStore(tmp_path / "empty")
    store.prepare()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        store.wait_ready(timeout_s=0.2)
    assert time.monotonic() - start >= 0.2


def test_validate_bundle_rejects_digest_and_model_path() -> None:
    tensors = [_meta("weight")]
    digest = compute_name_digest(tensors)
    bad_digest = WeightIpcBundle(
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
        validate_bundle(bad_digest, model_path="m", model_revision=None)

    bad_path = WeightIpcBundle(
        schema_version=SCHEMA_VERSION,
        model_path="a",
        model_revision=None,
        cuda_driver=None,
        created_unix_s=1.0,
        leader_pid=1,
        tensors=tensors,
        name_digest=digest,
    )
    with pytest.raises(WeightIpcImportError, match="model_path"):
        validate_bundle(bad_path, model_path="b", model_revision=None)
