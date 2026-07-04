# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize model runner for batched prefill audio embedding."""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner

logger = logging.getLogger(__name__)


class MossTranscribeDiarizeModelRunner(ModelRunner):
    """Runs MOSS-TD prefill with audio embeddings built across the prefill batch."""

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        if not any(
            getattr(
                getattr(sched_req.data.req, "multimodal_inputs", None),
                "mm_items",
                None,
            )
            for sched_req in requests
        ):
            return None

        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        model_dtype = next(self.model.language_model.parameters()).dtype

        all_audio_items = []
        for sched_req in requests:
            req = sched_req.data.req
            mm_inputs = getattr(req, "multimodal_inputs", None)
            all_audio_items.extend(list(getattr(mm_inputs, "mm_items", []) or []))

        audio_row_indices, audio_embed_indices = self._audio_indices(
            forward_batch,
            requests,
        )
        embed_input_ids = forward_batch.input_ids
        if audio_row_indices.numel() > 0:
            embed_input_ids = embed_input_ids.clone()
            embed_input_ids[audio_row_indices] = 0
        input_embeds = self.model.get_input_embeddings()(embed_input_ids).to(
            dtype=model_dtype
        )

        if not all_audio_items or audio_row_indices.numel() == 0:
            if all_audio_items:
                logger.debug(
                    "[moss-td] prefill_audio_batch reqs=%d items=%d audio_rows=0 "
                    "skip_encoder=True",
                    len(requests),
                    len(all_audio_items),
                )
            return input_embeds

        audio_pieces = self.model._encode_audio_items_batched(
            all_audio_items,
            forward_batch,
        )
        if not audio_pieces:
            return input_embeds
        audio_embeds = torch.cat(audio_pieces, dim=0).to(
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        if audio_embed_indices.numel() != audio_row_indices.numel():
            raise RuntimeError(
                "MOSS-TD prefill audio row/embed indices must align: "
                f"got {audio_row_indices.numel()} rows and "
                f"{audio_embed_indices.numel()} embed indices"
            )
        if (
            audio_embed_indices.numel() > 0
            and int(audio_embed_indices.max().item()) >= audio_embeds.shape[0]
        ):
            raise RuntimeError(
                "MOSS-TD prefill audio embed indices exceed available audio embeds: "
                f"got max index {int(audio_embed_indices.max().item())} for "
                f"{audio_embeds.shape[0]} embeds"
            )
        input_embeds[audio_row_indices] = audio_embeds[audio_embed_indices]

        logger.debug(
            "[moss-td] prefill_audio_batch reqs=%d items=%d audio_rows=%d",
            len(requests),
            len(all_audio_items),
            int(audio_row_indices.numel()),
        )
        return input_embeds

    @staticmethod
    def _audio_indices(
        forward_batch: Any,
        requests: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_indices: list[int] = []
        embed_indices: list[int] = []
        row_start = 0
        audio_embed_start = 0
        for sched_req in requests:
            req = sched_req.data.req
            prefix_len = len(req.prefix_indices)
            extend_len = int(req.extend_input_len)
            row_end = row_start + extend_len
            mm_inputs = getattr(req, "multimodal_inputs", None)
            audio_items = list(getattr(mm_inputs, "mm_items", []) or [])
            for item in audio_items:
                pad_value = int(getattr(item, "pad_value"))
                for start, end in getattr(item, "offsets", []):
                    original_start = int(start)
                    original_end = int(end)
                    clipped_start = max(original_start, prefix_len)
                    clipped_end = min(original_end, prefix_len + extend_len - 1)
                    if clipped_end < clipped_start:
                        audio_embed_start += original_end - original_start + 1
                        continue
                    rel_start = clipped_start - prefix_len
                    rel_end = clipped_end - prefix_len + 1
                    local_ids = forward_batch.input_ids[
                        row_start + rel_start : row_start + rel_end
                    ]
                    matches = torch.nonzero(
                        local_ids == pad_value,
                        as_tuple=False,
                    ).flatten()
                    if matches.numel() == 0:
                        audio_embed_start += original_end - original_start + 1
                        continue
                    row_indices.extend(
                        (row_start + rel_start + matches).detach().cpu().tolist()
                    )
                    embed_indices.extend(
                        (
                            audio_embed_start
                            + (clipped_start - original_start)
                            + matches
                        )
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    audio_embed_start += original_end - original_start + 1
            row_start = row_end

        if row_start != int(forward_batch.input_ids.shape[0]):
            raise RuntimeError(
                "MOSS-TD prefill request lengths must align with forward input_ids: "
                f"requests cover {row_start} rows for "
                f"{int(forward_batch.input_ids.shape[0])} input ids"
            )
        return (
            torch.tensor(
                row_indices,
                device=forward_batch.input_ids.device,
                dtype=torch.long,
            ),
            torch.tensor(
                embed_indices,
                device=forward_batch.input_ids.device,
                dtype=torch.long,
            ),
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.language_model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        logits_output = self.model(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_embeds_are_projected=True,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
