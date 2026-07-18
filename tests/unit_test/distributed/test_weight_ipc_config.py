# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.distributed.weight_ipc.config import (
    apply_weight_ipc_cli_env,
    is_weight_ipc_follower,
    resolve_weight_ipc_config,
)
from sglang_omni.distributed.weight_ipc.types import WeightIpcConfig


def test_resolve_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_OMNI_WEIGHT_IPC_ROLE", raising=False)
    monkeypatch.delenv("WEIGHT_IPC_ROLE", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_WEIGHT_IPC_STORE", raising=False)
    monkeypatch.delenv("WEIGHT_IPC_STORE", raising=False)
    cfg = resolve_weight_ipc_config()
    assert cfg.role == "off"
    assert not is_weight_ipc_follower()


def test_resolve_leader_requires_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHT_IPC_ROLE", "leader")
    monkeypatch.delenv("WEIGHT_IPC_STORE", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_WEIGHT_IPC_STORE", raising=False)
    with pytest.raises(ValueError, match="store"):
        resolve_weight_ipc_config()


def test_resolve_follower_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("WEIGHT_IPC_ROLE", "follower")
    monkeypatch.setenv("WEIGHT_IPC_STORE", str(tmp_path))
    cfg = resolve_weight_ipc_config()
    assert cfg.role == "follower"
    assert cfg.store_dir == str(tmp_path)
    assert is_weight_ipc_follower()


def test_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("WEIGHT_IPC_ROLE", "follower")
    monkeypatch.setenv("WEIGHT_IPC_STORE", str(tmp_path / "env"))
    cfg = resolve_weight_ipc_config(
        WeightIpcConfig(role="leader", store_dir=str(tmp_path / "explicit"))
    )
    assert cfg.role == "leader"
    assert cfg.store_dir == str(tmp_path / "explicit")


def test_apply_cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("SGLANG_OMNI_WEIGHT_IPC_ROLE", raising=False)
    apply_weight_ipc_cli_env(role="leader", store=str(tmp_path), timeout_s=30.0)
    cfg = resolve_weight_ipc_config()
    assert cfg.role == "leader"
    assert cfg.store_dir == str(tmp_path)
    assert cfg.timeout_s == 30.0
