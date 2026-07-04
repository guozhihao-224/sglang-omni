# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph runner for MOSS-Transcribe-Diarize Whisper encoder windows."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class _CapturedEncoderGraph(NamedTuple):
    """One captured per-shape graph and its static replay buffers."""

    graph: torch.cuda.CUDAGraph
    static_input_features: torch.Tensor
    static_encoder_position_ids: torch.Tensor
    static_output: torch.Tensor
    batch_size: int
    feature_len: int


class MossTranscribeDiarizeEncoderCudaGraphRunner:
    """Lazy-captured CUDA graphs for fixed-window Whisper encoder batches.

    Graphs are keyed by ``(batch_size, feature_len)``. A replay copies the live
    input features into the captured static buffer, replays the graph, and returns
    the static output buffer. Callers must consume/copy the result before a later
    replay of the same shape overwrites it.
    """

    def __init__(
        self,
        whisper_encoder,
        *,
        batch_sizes: Iterable[int] = (1, 2, 4, 8, 16, 32),
        max_graphs: int = 32,
        warmup_iters: int = 3,
        min_free_gb: float = 1.0,
    ) -> None:
        self._whisper_encoder = whisper_encoder
        self._batch_sizes = tuple(
            sorted({int(bs) for bs in batch_sizes if int(bs) > 0})
        )
        self._max_graphs = int(max_graphs)
        self._warmup_iters = int(warmup_iters)
        self._min_free_bytes = int(float(min_free_gb) * 1024**3)
        self._graphs: dict[tuple[int, int], _CapturedEncoderGraph] = {}
        self._failed_shapes: set[tuple[int, int]] = set()
        self._pool = None

    def captured_shapes(self) -> list[tuple[int, int]]:
        return sorted(self._graphs.keys())

    def _bucket_batch_size(self, batch_size: int) -> int | None:
        for bucket in self._batch_sizes:
            if batch_size <= bucket:
                return bucket
        return None

    def _device(self) -> torch.device:
        return next(self._whisper_encoder.parameters()).device

    def _enough_free_vram(self, device: torch.device) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(device)
        return free >= self._min_free_bytes, free

    @torch.no_grad()
    def _capture_shape(
        self,
        *,
        batch_size: int,
        feature_len: int,
        encoder_position_ids: torch.Tensor,
        forward_batch,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if len(self._graphs) >= self._max_graphs:
            logger.warning(
                "MOSS-TD encoder CG cap %d reached; serving eager for B=%d T=%d",
                self._max_graphs,
                batch_size,
                feature_len,
            )
            return

        enough, free = self._enough_free_vram(device)
        if not enough:
            logger.warning(
                "MOSS-TD encoder CG: free VRAM %.1fGB < %.1fGB headroom; "
                "eager for B=%d T=%d",
                free / 1024**3,
                self._min_free_bytes / 1024**3,
                batch_size,
                feature_len,
            )
            return

        n_mels = int(getattr(self._whisper_encoder.config, "num_mel_bins", 80))
        static_input_features = torch.zeros(
            (batch_size, n_mels, feature_len), dtype=dtype, device=device
        )
        static_encoder_position_ids = encoder_position_ids.to(
            device=device, dtype=torch.long
        ).clone()

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._whisper_encoder(
                    static_input_features,
                    static_encoder_position_ids,
                    forward_batch,
                )
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            static_output = self._whisper_encoder(
                static_input_features,
                static_encoder_position_ids,
                forward_batch,
            )

        self._graphs[(batch_size, feature_len)] = _CapturedEncoderGraph(
            graph=graph,
            static_input_features=static_input_features,
            static_encoder_position_ids=static_encoder_position_ids,
            static_output=static_output,
            batch_size=batch_size,
            feature_len=feature_len,
        )
        logger.info(
            "Captured MOSS-TD encoder CUDA graph B=%d T=%d -> %s (%d cached)",
            batch_size,
            feature_len,
            tuple(static_output.shape),
            len(self._graphs),
        )

    @torch.no_grad()
    def encode(
        self,
        input_features: torch.Tensor,
        encoder_position_ids: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor | None:
        if not input_features.is_cuda:
            return None
        if input_features.dim() != 3:
            return None

        live_batch_size = int(input_features.shape[0])
        feature_len = int(input_features.shape[-1])
        batch_size = self._bucket_batch_size(live_batch_size)
        if batch_size is None:
            return None

        shape = (batch_size, feature_len)
        entry = self._graphs.get(shape)
        if entry is None:
            if shape in self._failed_shapes:
                return None
            try:
                with torch.cuda.device(input_features.device):
                    self._capture_shape(
                        batch_size=batch_size,
                        feature_len=feature_len,
                        encoder_position_ids=encoder_position_ids,
                        forward_batch=forward_batch,
                        dtype=input_features.dtype,
                        device=input_features.device,
                    )
            except Exception as exc:
                self._graphs.pop(shape, None)
                self._failed_shapes.add(shape)
                logger.warning(
                    "MOSS-TD encoder CG capture failed for B=%d T=%d: %s; "
                    "serving eager",
                    batch_size,
                    feature_len,
                    exc,
                )
                return None
            entry = self._graphs.get(shape)
            if entry is None:
                self._failed_shapes.add(shape)
                return None

        entry.static_input_features.zero_()
        entry.static_input_features[:live_batch_size].copy_(input_features)
        entry.graph.replay()
        return entry.static_output[:live_batch_size]
