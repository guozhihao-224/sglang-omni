# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from sglang_omni.distributed.weight_ipc.types import WeightIpcConfig
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)


def _runner_with_config(cfg: WeightIpcConfig) -> SGLModelRunner:
    runner = object.__new__(SGLModelRunner)
    runner._weight_ipc_config = cfg
    runner.server_args = SimpleNamespace(
        model_path="m",
        revision="r1",
        load_format="auto",
    )
    runner.model = _Tiny()
    return runner


def test_load_model_off_calls_super(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner_with_config(WeightIpcConfig(role="off"))
    calls: list[str] = []

    def fake_super_load(self: Any) -> None:
        calls.append("super")

    monkeypatch.setattr(
        SGLModelRunner.__bases__[0],
        "load_model",
        fake_super_load,
    )
    SGLModelRunner.load_model(runner)
    assert calls == ["super"]


def test_load_model_leader_exports(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    runner = _runner_with_config(
        WeightIpcConfig(role="leader", store_dir=str(tmp_path))
    )
    calls: list[str] = []

    def fake_super_load(self: Any) -> None:
        calls.append("super")

    def fake_export(model, config, *, model_path, model_revision) -> None:
        calls.append(f"export:{model_path}:{model_revision}")
        assert model is runner.model
        assert config.role == "leader"

    monkeypatch.setattr(SGLModelRunner.__bases__[0], "load_model", fake_super_load)
    monkeypatch.setattr(
        "sglang_omni.model_runner.sglang_model_runner.export_leader_weights",
        fake_export,
    )
    SGLModelRunner.load_model(runner)
    assert calls == ["super", "export:m:r1"]


def test_load_model_follower_uses_dummy_then_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = _runner_with_config(
        WeightIpcConfig(role="follower", store_dir=str(tmp_path))
    )
    calls: list[str] = []

    def fake_super_load(self: Any) -> None:
        calls.append(f"super:{self.server_args.load_format}")

    def fake_materialize(model, config, *, model_path, model_revision):
        calls.append(f"materialize:{model_path}:{model_revision}")
        assert model is runner.model
        return MagicMock()

    monkeypatch.setattr(SGLModelRunner.__bases__[0], "load_model", fake_super_load)
    monkeypatch.setattr(
        "sglang_omni.model_runner.sglang_model_runner.materialize_follower_weights",
        fake_materialize,
    )
    SGLModelRunner.load_model(runner)
    assert calls == ["super:dummy", "materialize:m:r1"]
    assert runner.server_args.load_format == "auto"
