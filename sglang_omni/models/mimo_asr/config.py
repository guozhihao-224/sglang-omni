# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiMo-V2.5-ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.mimo_asr"


class MiMoASRPipelineConfig(PipelineConfig):
    """Single-stage ASR pipeline for MiMo-V2.5-ASR checkpoints."""

    architecture: ClassVar[str] = "MiMoV2ASRForCausalLM"

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_mimo_asr_executor",
            factory_args={
                "device": "cuda:0",
                "audio_tokenizer_path": "XiaomiMiMo/MiMo-Audio-Tokenizer",
                "max_running_requests": 8,
                "request_build_max_workers": 1,
                "request_build_max_pending": 8,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = MiMoASRPipelineConfig
