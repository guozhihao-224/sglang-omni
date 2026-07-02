# MiMo-V2.5-ASR Native SGLang-Omni Implementation Plan

## Goal

Support `XiaomiMiMo/MiMo-V2.5-ASR` in this repository with a **native SGLang-Omni backend**, not by wrapping the official high-level `MimoAudio.asr_sft()` API.

Use a **dual-track** integration model:

- **Pipeline / registration** (like `qwen3_asr` and PR #898 `fun_asr`): `PipelineConfig`, stage factory, `OmniScheduler` shell, request/result adapters, tests, cookbook.
- **Model + decode** (like official MiMo / vLLM-Omni): RVQ prefill embedding, `slm_sample` dual-channel decode, and likely a **custom `ModelRunner`** — not Qwen3-ASR's standard single-token AR path.

The target architecture is:

- A single ASR pipeline stage managed by SGLang-Omni.
- A custom SGLang-compatible model class registered into SGLang's `ModelRegistry`.
- A request builder that converts audio into MiMo audio codes and passes them through SGLang multimodal request data.
- A result adapter that returns transcription text in the same shape as existing ASR stages.

Non-goals for the first implementation:

- TTS / code2wav / speech output support.
- Full MiMo-Audio omni pipeline support.
- Realtime streaming transcription.
- Aggressive batching, CUDA graph, or multi-GPU optimization before single-request correctness is proven.

## Current Reference: How `qwen3_asr` Works

`qwen3_asr` is the reference for **pipeline wiring only** (config, stage, registry, scheduler adapters). It is **not** a reference for MiMo decode semantics.

Key files:

- `sglang_omni/models/qwen3_asr/config.py`
- `sglang_omni/models/qwen3_asr/stages.py`
- `sglang_omni/models/qwen3_asr/request_builders.py`
- `sglang_omni/models/qwen3_asr/sglang_model.py`
- `sglang_omni/model_runner/sglang_model_runner.py`
- `sglang_omni/model_runner/model_worker.py`

The `qwen3_asr` flow is:

1. `Qwen3ASRPipelineConfig` declares `architecture = "Qwen3ASRForConditionalGeneration"`.
2. `create_sglang_qwen3_asr_executor()` starts SGLang infrastructure with `model_arch_override="Qwen3ASRForConditionalGeneration"`.
3. `ModelWorker` overrides `model_config.hf_config.architectures` to this architecture.
4. `SGLModelRunner._register_omni_model()` registers `Qwen3ASRForConditionalGeneration` into SGLang's `ModelRegistry`.
5. `sglang_model.py` defines the actual model class that SGLang loads.
6. `request_builders.py` converts audio to model inputs and stores multimodal data on `Req`.
7. The model `forward()` injects audio embeddings into placeholder token positions.
8. `result_adapter()` decodes generated token ids into ASR text.

The MiMo implementation should mirror this **pipeline control flow**; model internals differ (see below).

## Important Difference From `qwen3_asr`

`qwen3_asr` audio path:

```text
audio waveform
  -> AutoFeatureExtractor / Whisper-style features
  -> Qwen3 audio_tower
  -> audio embeddings
  -> injected into <|audio_pad|> positions
  -> Qwen3 causal LM generates transcript
```

MiMo-V2.5-ASR audio path:

```text
audio waveform
  -> MiMo-Audio-Tokenizer (24 kHz mel -> RVQ)
  -> [T, 8] audio codes (padded to multiple of group_size=4)
  -> speech_embeddings + input_local_transformer + speech_group_downcast
  -> merged into <|empty|> positions (1 empty token per 4 code frames)
  -> 36L Qwen2 causal LM + slm_sample decode loop
  -> transcript (text channel, stride group_size)
```

Therefore MiMo reuses the same SGLang-Omni **pipeline registration** pattern as `qwen3_asr`, but cannot reuse its model class, request builder, default `ModelRunner`, or standard single-token decode path.

## Architecture Decision: Dual-Channel Decode (`slm_sample`)

**Decision:** MiMo-ASR native integration is **not** “prefill inject audio + standard SGLang sampling.” It requires reproducing official **`slm_sample`**: one **logical decode group** per scheduler step may append **`(audio_channels + 1) × group_size = 36` flat tokens**, plus optional `local_forward` when the global LM emits `<|empty|>`.

Official loop (`modeling_mimo_audio.py`):

```text
1. Global LM predicts next text token (greedy for ASR).
2. If token == <|empty|> (151667):
     local_transformer generates 8 x group_size RVQ codes via local_forward.
   Else:
     speech channels fill speech_zeroemb indices.
3. Append (text + 8 speech channels) x group_size to the flat sequence.
4. Stop when MiMoStopper sees <|im_end|> (151645) in the text slot.
```

**Implications (may span more than `sglang_model.py`):**

| Layer | Responsibility |
|-------|----------------|
| `sglang_model.py` | `_prepare_input_embeds`, `local_forward`, logits / hidden states |
| **`ModelRunner` subclass** (likely) | After each forward: global sample + local branch + write 36 flat tokens into `req` / KV; see Phase 5 |
| `OmniScheduler` | May stay as shell **if** custom runner satisfies the one-step contract; otherwise extend (Phase 5 Option B) |
| Request builder | Build 9-row flat prompt + attach RVQ codes; not Qwen3 mel features |

Reference priority:

| Concern | Primary reference | Secondary reference |
|---------|-------------------|---------------------|
| Pipeline / tests / cookbook | PR #898 `fun_asr` | `qwen3_asr` |
| Prefill embedding + decode loop | Official `modeling_mimo_audio.py` | vLLM-Omni `mimo_audio_llm.py` |
| Custom runner precedent | `sglang_omni/models/moss_tts_local/model_runner.py` | — |
| Prompt / tokenizer front-end | Official `mimo_audio.py`, `process_speechdata.py` | — |

### Flat Sequence Layout

MiMo stores prompt/state as **`[9, L]`** before flattening: row 0 = text channel, rows 1–8 = RVQ channels. Each **logical token group** occupies **`group_size=4`** consecutive columns across all 9 rows.

**Example:** one text token `T` and one `<|empty|>` audio group with RVQ codes `(c0..c7)` per sub-step:

```text
Rows (channel):     text   rvq0  rvq1  ...  rvq7
Col group (x4):     [ T    c0..  c0..       c7.. ]  -> 4 columns per group
                    [ -    c0..  c0..       c7.. ]
                    [ -    c0..  c0..       c7.. ]
                    [ -    c0..  c0..       c7.. ]

Flat index (conceptual): 9 rows x 4 cols = 36 ids per group, concatenated in official order.
Decode append: one slm_sample iteration adds one new group (36 flat ids).
Transcript: read text row at flat indices 0, 36, 72, ... i.e. stride (9 * group_size) in flattened layout,
            or equivalently text_row[::group_size] after reshape to [9, L].
```

Implementers must not confuse:

- **Text-only `input_ids` list** used in some SGLang APIs vs **9-channel flat tensor** used inside MiMo.
- **Placeholder count** (`L_text / 4` empty tokens in audio segment) vs **flat length** (`L_text * 4 * 9` in flattened storage).

### Token Unit Glossary

| Term | Meaning | Example |
|------|---------|---------|
| **Code frame** | One RVQ timestep | `[8]` codes |
| **Group** | `group_size` code frames | 4 frames → 1 text-slot in audio segment |
| **Logical decode step** | One official `slm_sample` iteration | `cur_len += 1` in official code |
| **Flat tokens appended per step** | `(audio_channels+1) × group_size` | **36** |
| **`max_new_tokens` (official)** | Max **logical decode groups** after prompt | default **8192** |
| **`SamplingParams.max_new_tokens` (SGLang)** | Usually **scheduler forward steps** or output token count | **Must be mapped explicitly** — do not pass 8192 blindly |

**Context length planning:**

```text
max_flat_tokens ≈ prompt_flat_len + max_new_tokens_groups * 36
max_kv_groups   ≈ prompt_groups + max_new_tokens_groups
```

Stage `context_length` and `max_prefill_tokens` should use **flat** or **group** units consistently; document the conversion in `stages.py`.



## Additional Reference: SGLang-Omni PR #898

Also use `https://github.com/sgl-project/sglang-omni/pull/898` as an implementation reference.

PR #898 adds a native SGLang-Omni ASR backend for Fun-ASR-Nano. It is not MiMo-ASR, but it is highly relevant because it follows the same repository-level integration style we want for MiMo:

- A new model package under `sglang_omni/models/fun_asr/`.
- A `PipelineConfig` with architecture aliases.
- A custom `sglang_model.py` registered into SGLang's model registry.
- A stage factory that creates SGLang infrastructure with `model_arch_override`.
- A request builder and result adapter with ASR-compatible output schema.
- Unit tests for config resolution, stage defaults, request construction, and result decoding.
- A cookbook page and benchmark updates that make the ASR backend selectable.

Files from PR #898 to compare against while implementing MiMo:

```text
sglang_omni/models/fun_asr/config.py
sglang_omni/models/fun_asr/stages.py
sglang_omni/models/fun_asr/request_builders.py
sglang_omni/models/fun_asr/sglang_model.py
sglang_omni/models/fun_asr/configuration_fun_asr.py
sglang_omni/models/fun_asr/tool_funcs/audio_lengths.py
tests/unit_test/fun_asr/test_pipeline.py
tests/unit_test/fun_asr/test_request_builders.py
docs/cookbook/fun_asr.md
```

Concrete patterns to reuse:

1. **Architecture aliases**: Fun-ASR registers multiple possible HF architecture names. MiMo should do the same if the released checkpoint, official code, and reference implementations use slightly different names.
2. **Custom HF configuration module**: Fun-ASR includes `configuration_fun_asr.py` to register config/processor behavior that the upstream HF stack does not provide directly. MiMo may need `configuration_mimo_asr.py` for the same reason.
3. **Stage defaults**: Fun-ASR uses a single terminal ASR stage with explicit `max_running_requests`, dtype, context-length estimation, and SGLang server args. MiMo should start similarly but with more conservative concurrency.
4. **Length helper module**: Fun-ASR adds `tool_funcs/audio_lengths.py` to compute audio-token length from feature length. MiMo should add `tool_funcs/audio_lengths.py` with `num_code_frames -> num_empty_tokens` (see Phase 0).
5. **Tests first for routing/config**: Fun-ASR adds unit tests that do not require a full model load. MiMo should add the same class of tests early.
6. **Cookbook after backend works**: Fun-ASR documents launch, API usage, parameters, and benchmark usage. MiMo should follow that structure after smoke tests pass.

Important distinction:

- PR #898's model internals are Fun-ASR-specific encoder/decoder logic.
- MiMo should reuse PR #898's integration pattern, tests, and documentation shape, but MiMo's `sglang_model.py` must implement MiMo audio-code embedding and causal LM injection, not Fun-ASR's encoder path.

## External References To Study

Primary model assets:

- `XiaomiMiMo/MiMo-V2.5-ASR`
- `XiaomiMiMo/MiMo-Audio-Tokenizer`

Reference implementation sources:

- Official `XiaomiMiMo/MiMo-V2.5-ASR` repository and README.
- vLLM-Omni MiMo implementation, especially:
  - `mimo_audio.py`
  - `mimo_audio_llm.py`
  - `config_mimo_audio.py`

These items were resolved in Phase 0 (see below). Keep the official repo and vLLM-Omni sources open while implementing `sglang_model.py` and the decode loop.

## Proposed File Layout

Add a new package:

```text
sglang_omni/models/mimo_asr/
  __init__.py
  config.py
  stages.py
  request_builders.py
  sglang_model.py
  configuration_mimo_asr.py   # MiMoAudioConfig extends Qwen2Config
  audio_tokenizer.py          # MiMo-Audio-Tokenizer adapter
  prompt.py                   # InputSegment + get_asr_sft_prompt port
  process_speechdata.py       # optional: port official InputSegment helpers
  model_runner.py             # MiMo-specific ModelRunner (decode integration)
  tool_funcs/
    audio_lengths.py          # num_code_frames -> num <|empty|> tokens

tests/unit_test/mimo_asr/
  test_pipeline.py
  test_request_builders.py
  test_parity.py              # prompt / embedding / decode parity vs official

docs/cookbook/mimo_asr.md     # after smoke passes
examples/run_mimo_asr_server.py
tests/test_model/test_mimo_asr_ci.py
```

## Phase 0: Research And Shape Confirmation

Status: **complete** (2026-06-30). Sources: HF `XiaomiMiMo/MiMo-V2.5-ASR` config/tokenizer, official GitHub repo, vLLM-Omni #3089.

### Checkpoint And Architecture

| Field | Confirmed value |
|-------|-----------------|
| `architectures[0]` | `MiMoV2ASRForCausalLM` |
| `model_type` | `qwen2` (flat top-level config, **no** nested `llm_config` / `thinker_config`) |
| Total parameters | ~7.62B |
| Text backbone | 36-layer Qwen2: `hidden_size=4096`, `num_attention_heads=32`, `num_key_value_heads=8`, `intermediate_size=11008`, `rope_theta=640000`, `max_position_embeddings=8192` |
| `vocab_size` | 151680 |
| Audio | `audio_channels=8`, `group_size=4`, `input_local_layers=6`, `input_local_dim=1024`, `input_full_attention=true` |
| RVQ | `speech_vocab_size="1025-1025-129-129-129-129-129-129"`, `speech_zeroemb_idx="1024-1024-128-128-128-128-128-128"` |
| `delay_pattern` | `"0-1-2-3-4-5-6-7"` |
| RoPE | Ordinary RoPE (`rope_scaling=null`); **not** MRoPE for ASR |

**Phase 3 implication:** do **not** add `_ARCH_CONFIG_MAP` for MiMo. SGLang memory planning can use the flat Qwen2 fields in the checkpoint config directly.

### Special Token IDs

| Token | ID | ASR role |
|-------|-----|----------|
| `<\|empty\|>` | **151667** | Audio placeholder / empty decode step |
| `<\|redacted_im_end\|>` (eos) | **151645** | **Stop token** |
| `<\|im_start\|>` | 151644 | Chat turn marker |
| `<\|sosp\|>` / `<\|eosp\|>` | 151665 / 151666 | Audio segment boundaries in speech channels |
| `<\|SpeechLM\|>` | 151669 | Assistant prefix |
| `<\|eot\|>` | 151672 | Inside assistant prefix; strip from output |
| `<\|sostm\|>` / `<\|eostm\|>` | 151670 / 151671 | Streaming TTS; strip `<\|eostm\|>` from ASR output |
| `<\|endoftext\|>` (pad) | 151643 | Padding |
| `<chinese>` / `<english>` | plain text `audio_tag` strings | Language bias (not special tokens) |

### MiMo-Audio-Tokenizer

| Item | Value |
|------|-------|
| Sample rate | 24000 Hz |
| Mel | `n_mels=128`, `nfft=960`, `hop_length=240`, `window_size=960` |
| Mono | Stereo input is mean-reduced to mono |
| Chunking | 30 s chunks; tail shorter than `n_fft` merged or zero-padded |
| Encoder output | `[T, 8]` int codes (first 8 of 20 quantizer layers) |
| Padding | Pad frame count to multiple of `group_size=4` (repeat last frame) |
| Runtime | Official path uses GPU bfloat16; keep `request_build_max_workers=1` initially |

### Placeholder / Length Mapping

- **Not** 1 code frame = 1 `<\|empty\|>`.
- **`num_empty_tokens = ceil(num_code_frames / group_size)`** after padding.
- Text tokens outside the audio segment expand with `insert_between(..., group_size-1)` (one effective token per group of 4 positions).
- Flat sequence step size per generated group: `(audio_channels + 1) * group_size = 36`.

```python
def mimo_asr_num_empty_tokens(num_code_frames: int, group_size: int = 4) -> int:
    padded = ((num_code_frames + group_size - 1) // group_size) * group_size
    return padded // group_size
```

### ASR Prompt (`get_asr_sft_prompt`)

Six `InputSegment` blocks, in order:

1. `"<\|im_start\|>user\n"`
2. Audio codes (`add_sosp_eosp=True` on speech channels)
3. Random line from `asr_zh_templates` or `asr_en_templates`
4. `"<\|redacted_im_end\|>\n"`
5. `"<\|im_start\|>assistant\n"`
6. `"<\|SpeechLM\|>\n\n<\|eot\|>\n{audio_tag}"`

`audio_tag` selection:

| Tag | Template pool |
|-----|----------------|
| `"<chinese>"` | `asr_zh_templates` |
| `"<english>"` | `asr_en_templates` |
| `""` (Auto) | zh + en pools |

Post-decode cleanup: strip `<\|empty\|>`, `<\|eot\|>`, `<\|eostm\|>`, and `<chinese>` / `<english>` if present.

### Sampling And Stop (ASR task)

| Sampler | Official ASR setting |
|---------|----------------------|
| Global (text LM) | `do_sample=False` (greedy) |
| Local (on `<\|empty\|>` steps) | `do_sample=True`, `temperature=0.9`, `top_p=0.95` |

**Determinism:** Global text is greedy and reproducible. Local sampling is stochastic unless seeded. For **transcript parity** vs official `asr_sft()`, compare decoded text (WER), not necessarily token-id parity. For **token-id parity** tests, fix `local_forward` RNG seed or use greedy local sampling in test-only mode.

Stop: `MiMoStopper(stop_tokens=[151645])` — checks the text-channel slot every 36 flat tokens.

`max_new_tokens`: official default **8192 logical groups** (see Token Unit Glossary). Map explicitly to SGLang `SamplingParams` and `context_length`; do not assume SGLang counts groups.

Note: HF `generation_config.json` (`temperature=0.6`, `top_p=0.95`) is overridden by the ASR task sampler config above.

### Checkpoint Weight Prefixes

Derive required prefixes from `model.safetensors.index.json` (and fail load if any required prefix is missing). The current ASR release contains **only** ASR modules (~7.62B); no TTS/code2wav/vocoder keys.

Maintain an explicit **skip allowlist** for optional keys absent from this checkpoint (e.g. future `code2wav*`, `vocoder*`). Never skip `local_transformer*` for the current ASR release.

| Prefix | Load? | Role |
|--------|-------|------|
| `model.*` | Yes | 36L Qwen2 LM → SGLang `Qwen2ForCausalLM` |
| `lm_head.*` | Optional | Current ASR checkpoint omits this prefix; load it if present |
| `speech_embeddings.{0-7}.*` | Yes | 8-channel code embeddings |
| `input_local_transformer.*` | Yes | Prefill audio re-encoding |
| `speech_group_downcast.*` | Yes | Group → hidden |
| `hidden_states_downcast.*` | Yes | LM hidden → local dim |
| `local_transformer.*` | **Yes** | Required when decode predicts `<\|empty\|>` |
| `local_transformer_lm_heads.{0-7}.*` | **Yes** | Same |

Apply SGLang qkv / gate_up fusion when loading `model.*`, `input_local_transformer.*`, and `local_transformer.*`.

### Remaining Local Verification (Implementation Phase)

- Byte-level prompt token id parity vs `MimoAudio.get_asr_sft_prompt()` on a fixed WAV.
- Tune `max_new_tokens` / `max_prefill_tokens` after first GPU smoke.
- Confirm PR #898 `fun_asr` is merged on the target branch before mirroring its test layout.

## Phase 1: Pipeline Config

Create `sglang_omni/models/mimo_asr/config.py`.

Expected structure:

```python
from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.mimo_asr"


class MiMoASRPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MiMoV2ASRForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = ()

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
```

Notes:

- Start with conservative defaults because audio tokenization and embedding can be memory-heavy.
- `architecture = "MiMoV2ASRForCausalLM"` matches the released checkpoint; `architecture_aliases` can stay empty unless a future alias appears.
- Keep `audio_tokenizer_path` configurable through stage `factory_args`.

## Phase 2: SGLang Registry Integration

Modify `sglang_omni/model_runner/sglang_model_runner.py`.

Add to the `sglang_omni_models` mapping in `_register_omni_model()`:

```python
"MiMoV2ASRForCausalLM": "sglang_omni.models.mimo_asr.sglang_model:MiMoV2ASRForCausalLM",
```

This mirrors the existing `Qwen3ASRForConditionalGeneration` registration.

## Phase 3: Model Config Override

**No change required** for `sglang_omni/model_runner/model_worker.py`.

Unlike `Qwen3ASRForConditionalGeneration`, the MiMo-V2.5-ASR checkpoint uses a **flat** Qwen2-shaped `config.json` (`model_type: qwen2`). There is no nested `thinker_config.text_config` or `llm_config` subtree.

Do **not** add a `_ARCH_CONFIG_MAP` entry for `MiMoV2ASRForCausalLM`. When the map lookup misses, SGLang already reads `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `num_key_value_heads` from the top-level HF config, which is correct for this model.

Acceptance criteria:

- After `model_arch_override="MiMoV2ASRForCausalLM"`, memory pool sizing uses `hidden_size=4096`, `num_hidden_layers=36`, `num_attention_heads=32`, `num_key_value_heads=8`.
- No incorrect nested-config override is applied.
- **Smoke check:** log or assert `ModelConfig` fields after override (`hidden_size`, `num_hidden_layers`, `vocab_size`) match Phase 0 — guard against silent mis-read when `hf_text_config` is unset.

## Phase 4: Stage Factory

Create `sglang_omni/models/mimo_asr/stages.py`.

Responsibilities:

1. Load tokenizer with `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`.
2. Initialize MiMo audio tokenizer or a lightweight wrapper object.
3. Build SGLang server args with conservative defaults.
4. Call `create_sglang_infrastructure_defer_cuda_graph()` with `model_arch_override="MiMoV2ASRForCausalLM"`.
5. Initialize multimodal embedding cache if needed.
6. Build request/result adapters.
7. Wire **`MiMoASRModelRunner`** (or default `ModelRunner` only after Phase 5 Option A/C validation).
8. Return `OmniScheduler`.

**Scheduler compatibility note:** Qwen3-ASR uses default `ModelRunner(model_worker, output_proc)` with standard `_sample_next_token_ids`. MiMo likely needs a **custom runner** (see Phase 5; precedent: `moss_tts_local/model_runner.py`). Do not assume `create_sglang_infrastructure_defer_cuda_graph()` + stock `OmniScheduler` is sufficient until Phase 5 Option C is validated on one request.

Skeleton:

```python
def create_sglang_mimo_asr_executor(
    model_path: str,
    *,
    audio_tokenizer_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 8,
    max_new_tokens_groups: int = 8192,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    request_build_max_workers: int = 1,
    request_build_max_pending: int | None = 8,
    server_args_overrides: dict[str, Any] | None = None,
):
    ...
```

Suggested initial server defaults:

```python
defaults = {
    "disable_cuda_graph": True,
    "disable_overlap_schedule": True,
    "enable_torch_compile": False,
    "mem_fraction_static": mem_fraction_static,
    "max_prefill_tokens": 8192,
    "chunked_prefill_size": 8192,
    "sampling_backend": "pytorch",
    "dtype": dtype,
}
```

Start with CUDA graph disabled until the embedding path is stable.

## Phase 5: Custom Decode Integration

**Goal:** Connect official `slm_sample` semantics to SGLang-Omni's `OmniScheduler` / `ModelRunner` contract.

Qwen3-ASR path (insufficient for MiMo):

```text
forward -> logits -> _sample_next_token_ids -> append 1 token -> repeat
```

MiMo path (required):

```text
forward -> global greedy text token
        -> if empty: local_forward (sampled RVQ) else speech_zeroemb fill
        -> append 36 flat tokens
        -> MiMoStopper
        -> repeat
```

### Scheduler Compatibility Plan

| Option | Description | When to use |
|--------|-------------|-------------|
| **A** | Hide multi-token append inside model/`ModelRunner.post_decode`; scheduler still advances one logical step per forward | If KV + `prepare_for_decode` can treat one group as one step |
| **B** | MiMo-specific scheduler or decode manager maintaining 9-channel flat state | If Option A cannot satisfy async decode / batching invariants |
| **C (recommended first)** | Single-request, non-overlap custom loop in **`MiMoASRModelRunner`**: sync `post_decode`, no batching, correctness first | **Phase 1 implementation** — matches non-goals |

**Recommended path:** **Option C** → validate transcript parity → evolve toward A for batching; defer B unless required.

Reference implementation in this repo: `sglang_omni/models/moss_tts_local/model_runner.py` (`post_decode`, `post_decode_launch`, `post_decode_resolve`, custom collect loop).

### `max_new_tokens` Mapping

Expose **`max_new_tokens_groups`** (default 8192) in stage/request config. When filling SGLang `SamplingParams`:

- Document whether the runner counts **logical groups** or **flat tokens**.
- Set `context_length` using flat-token estimate from Token Unit Glossary.
- Add a unit test that one decode step appends exactly **36** flat tokens for a non-empty batch.

### Acceptance Criteria

- One short WAV completes without scheduler assertion failures.
- First decode step matches official first-step text token (global greedy).
- Full transcript matches official `MimoAudio.asr_sft()` on smoke WAV (text parity; token parity optional).

## Phase 6: Custom SGLang Model

Create `sglang_omni/models/mimo_asr/sglang_model.py`.

This is the highest-risk phase. Use **qwen3_asr** for SGLang module shape and weight-fusion patterns, but port **prefill + decode** logic from official `modeling_mimo_audio.py` and vLLM-Omni `mimo_audio_llm.py`.

### Required Class

```python
class MiMoV2ASRForCausalLM(nn.Module):
    ...

EntryClass = MiMoV2ASRForCausalLM
```

### Expected Submodules

Load all prefixes present in the ASR checkpoint:

- Text causal LM: SGLang **`Qwen2ForCausalLM`** backed by checkpoint `model.*`.
- `speech_embeddings` (8 channels).
- `input_local_transformer` (6 layers, prefill re-encoding).
- `speech_group_downcast`, `hidden_states_downcast`.
- **`local_transformer`** (16 layers) and **`local_transformer_lm_heads`** — required for decode when the global LM emits `<|empty|>`; this is **not** TTS-only despite the name.
- `lm_head` is optional; the current ASR checkpoint omits this prefix.

Do **not** expect TTS/code2wav/vocoder weights in the ASR checkpoint. Derive required vs skippable keys from the checkpoint index; maintain a skip allowlist for absent optional prefixes only.

### Prefill: Input Embedding

Port `_prepare_input_embeds` from official code:

- Input layout: `[B, audio_channels+1, T*group_size]` (text row + 8 RVQ rows).
- For positions where text id == `<|empty|>` (151667): sum speech embeddings across 8 channels, run `input_local_transformer`, `speech_group_downcast`, add to text embed (zeroed at empty positions first).
- Request builder / multimodal path must supply audio codes aligned to `<|empty|>` positions (see vLLM `_overlay_audio_codes_by_prompt_pad_positions`).

### Decode: `slm_sample` Loop

Standard SGLang `_sample_next_token_ids` (one text token per step) is **insufficient**. Implement decode in **`MiMoASRModelRunner`** (Phase 5) calling into model helpers:

1. Run global LM; greedy sample next text token (ASR task).
2. If token == 151667: `hidden_states_downcast` → `local_forward` → 8×`group_size` RVQ tokens.
3. Else: fill speech rows with `speech_zeroemb_idx` per channel.
4. Append `(text + speech) × group_size` flat tokens; check `MiMoStopper` on text slot.
5. Extract transcript from text row with stride `group_size` (see Flat Sequence Layout); strip special tokens.

Reference vLLM-Omni `MiMoAudioLLMForConditionalGeneration.forward` for how to merge cached audio embeddings on decode steps where `input_ids == empty_token_id`.

### SGLang Hooks (naming note)

MiMo uses **RVQ codes → embeddings**, not mel features. Keep SGLang multimodal hook names where required, but implement MiMo semantics:

```python
def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
    """Expand placeholders / align <|empty|> count with audio codes."""

def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
    """SGLang hook name only — returns RVQ codes or drives prepare_multimodal_embeddings."""

def prepare_multimodal_embeddings(self, ...):
    """Preferred internal name for _prepare_input_embeds equivalent."""

def forward(self, input_ids, positions, forward_batch, **kwargs):
    ...

def load_weights(self, weights):
    ...
```

`configuration_mimo_asr.py` should define `MiMoAudioConfig(Qwen2Config)` matching official fields (`group_size`, `audio_channels`, `speech_vocab_size`, etc.).

### Weight Loading

Plan:

1. Log unmatched keys during development.
2. Map `model.*` → SGLang Qwen2 with qkv / gate_up fusion.
3. Load all **required** prefixes from checkpoint index; skip only keys on the allowlist for absent optional modules.
4. Assert required-key coverage in unit test or load-time check.
5. Compare output text with official `MimoAudio.asr_sft()` on the same audio.

Acceptance criteria:

- All checkpoint prefixes load without missing required keys.
- Prefill + decode path matches official greedy-text / sampled-local behavior.
- Short WAV returns text aligned with official reference on a smoke sample.

## Phase 7: Audio Tokenizer Adapter

Create `sglang_omni/models/mimo_asr/audio_tokenizer.py`.

Port the official front-end from `MimoAudio.preprocess_input`:

- Load `XiaomiMiMo/MiMo-Audio-Tokenizer` (24 kHz, mel params from tokenizer config).
- Resample to 24000 Hz; stereo → mono mean.
- 30 s chunking with tail merge / zero-pad rules from official code.
- `encoder.encode(..., return_codes_only=True)` → `[T, 8]` codes.
- Pad `T` to a multiple of `group_size=4`.
- Return codes in the shape expected by `InputSegment` / request builder.

Suggested API:

```python
class MiMoAudioTokenizerAdapter:
    def __init__(self, audio_tokenizer_path: str, device: str):
        ...

    def encode(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Return [T, 8] int64 codes after group_size padding."""
        ...
```

Keep `request_build_max_workers=1` until GPU tokenization is proven thread-safe.

## Phase 8: Request Builder

Create `sglang_omni/models/mimo_asr/request_builders.py`.

Responsibilities:

1. Extract audio input from `StagePayload`.
2. Load audio from path, URL, bytes, or memoryview.
3. Tokenize audio using MiMo audio tokenizer.
4. Build official ASR prompt.
5. Encode prompt into `input_ids`.
6. Attach audio codes to `MultimodalInputs` / `Req`.
7. Build sampling params.
8. Decode output ids into transcript.

Input extraction should follow existing ASR behavior:

```python
for key in ("audio_bytes", "bytes", "file"):
    ...
for key in ("audio_path", "path", "url"):
    ...
```

Request data class:

```python
@dataclass
class MiMoASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "auto"
    audio_tag: str | None = None
    engine_start_s: float = 0.0
```

Parameter mapping:

- `audio_tag` passes through if supplied.
- `language="zh"` or `language="chinese"` maps to `<chinese>`.
- `language="en"` or `language="english"` maps to `<english>`.
- `language="auto"` leaves the official default if supported.

Sampling defaults (match official ASR task):

- Global text: `temperature=0.0` (greedy), `top_p=1.0`.
- Local on `<|empty|>` steps: handled inside model (`temperature=0.9`, `top_p=0.95`, `do_sample=True`) — not exposed as public API initially.
- `max_new_tokens_groups`: default **8192** logical decode groups; map to runner/SGLang limits (see Phase 5).

Result decoding:

- Take text channel with stride **`group_size`** on the **text row** of generated 9-channel state (see Flat Sequence Layout).
- Strip `<|empty|>`, `<|eot|>`, `<|eostm|>`, `<chinese>`, `<english>`.

Result adapter should return the same fields as `qwen3_asr`:

```python
StagePayload(
    request_id=payload.request_id,
    request=payload.request,
    data={
        "text": text,
        "language": data.language,
        "duration_s": data.audio_duration_s,
        "asr_latency_s": engine_time_s,
        "usage": {"engine_time_s": engine_time_s},
        "modality": "text",
    },
)
```

## Phase 9: Prompt Template

Create `prompt.py` (and optionally port `process_speechdata.py` / `templates.py` from the official repo).

Confirmed behavior (see Phase 0):

- Not driven by HF `chat_template.jinja` alone; built from six `InputSegment` blocks in `get_asr_sft_prompt`.
- `audio_tag` appended in the final assistant segment after `<|SpeechLM|>\n\n<|eot|>\n`.
- Audio segment: `T` code frames → **`T/4`** `<|empty|>` tokens in the text row; speech rows carry RVQ codes with `<|sosp|>`/`<|eosp|>` padding groups.
- Random instruction template when `audio_tag` is Auto.

Unit tests should assert placeholder count == `mimo_asr_num_empty_tokens(len(codes))` and match official token ids on a fixture WAV when the tokenizer adapter is available.

## Phase 10: HTTP/API Compatibility

Existing OpenAI-compatible routes already pass audio metadata into requests. The native ASR stage should work with the same transcription path used by Qwen3-ASR.

Verify these paths:

- `/v1/audio/transcriptions`
- `/v1/chat/completions` with audio metadata, if expected for this project

If needed, add model-specific parameter forwarding for:

- `language`
- `audio_tag`
- `max_new_tokens`

Do not introduce MiMo-specific public API fields unless they are optional and backward compatible.

## Phase 11: Tests

Start with targeted tests and avoid broad CI until single-request correctness is stable.

### Unit Smoke Test

Add a minimal test that:

1. Creates a `StagePayload` with a local audio path.
2. Calls the request builder.
3. Verifies:
   - prompt ids are non-empty
   - audio codes are present
   - placeholder count matches audio embedding/code expectations

This can run without loading the full model if audio tokenizer can be mocked.

### Parity Tests (`tests/unit_test/mimo_asr/test_parity.py` + GPU smoke)

Layered parity vs official `MimoAudio.asr_sft()` on a fixed WAV:

| Stage | Assert |
|-------|--------|
| Prompt token parity | `input_ids` match official (fix random template seed) |
| Audio code parity | `[T, 8]` codes match official tokenizer path |
| Prefill embedding | shapes / non-zero mask at `<\|empty\|>` positions |
| First decode step | first global text token matches (greedy) |
| Full transcript | decoded text matches (primary gate); token ids optional |

### Integration Smoke Test

Add a GPU-marked test that:

1. Launches MiMo-ASR server.
2. Calls `/v1/audio/transcriptions` with a short wav.
3. Asserts non-empty text.

### Correctness Test

After smoke passes:

- Reuse the Qwen3-ASR WER testing pattern.
- Use a small known ASR dataset subset.
- Track WER, latency, throughput, RTF, and GPU memory.

## Phase 12: Documentation

Add `docs/cookbook/mimo_asr.md` after implementation is working.

Contents:

- Model download commands.
- Audio tokenizer path requirement.
- Server launch command.
- `/v1/audio/transcriptions` example.
- `audio_tag` / `language` usage.
- Known limitations.

Example launch shape:

```bash
sgl-omni serve \
  --model-path XiaomiMiMo/MiMo-V2.5-ASR \
  --extra-args asr.audio_tokenizer_path=XiaomiMiMo/MiMo-Audio-Tokenizer
```

The exact CLI override syntax should match the project's current config manager behavior.

## Implementation Order

Recommended order:

1. ~~Confirm MiMo config and prompt details.~~ (Phase 0 done)
2. Review PR #898 `fun_asr` + **`moss_tts_local/model_runner.py`** for decode integration patterns.
3. Add `configuration_mimo_asr.py`, `mimo_asr/config.py`, and `__init__.py`.
4. Add `tool_funcs/audio_lengths.py` and routing/config unit tests (`test_pipeline.py`).
5. Add registry entry in `sglang_model_runner.py` (skip `_ARCH_CONFIG_MAP`).
6. Implement audio tokenizer adapter (`audio_tokenizer.py`).
7. Port `prompt.py` / `InputSegment` helpers and request builder; add `test_request_builders.py`.
8. Add stage factory skeleton with `model_arch_override` (stock runner OK temporarily).
9. Add skeleton `MiMoV2ASRForCausalLM` + weight loading with required-key coverage check.
10. Implement prefill embedding path (`_prepare_input_embeds` equivalent).
11. **Implement Phase 5:** `MiMoASRModelRunner` Option C (single-request `slm_sample` loop).
12. Add parity tests (`test_parity.py`); smoke vs official `asr_sft()`.
13. Enable small batching / Option A hardening.
14. Add cookbook, GPU CI, and benchmark integration.

## Acceptance Criteria

Minimum acceptable implementation:

- `ConfigManager.from_model_path("XiaomiMiMo/MiMo-V2.5-ASR")` resolves to `MiMoASRPipelineConfig`.
- The server starts with `model_arch_override="MiMoV2ASRForCausalLM"`.
- SGLang loads `sglang_omni.models.mimo_asr.sglang_model:MiMoV2ASRForCausalLM`.
- **`ModelConfig` smoke:** logged dimensions match Phase 0 after arch override.
- **`MiMoASRModelRunner`** completes single-request decode (Phase 5 Option C).
- A short WAV request returns non-empty transcription text matching official smoke reference.
- Output schema matches existing ASR stages.
- Required checkpoint prefixes load; skip allowlist documented for absent optional keys.
- Decode uses greedy global + official local sampling (or fixed seed in parity tests).
- Single-request latency and memory usage are recorded.

Stretch acceptance:

- Batch size 2 to 4 works.
- `/v1/audio/transcriptions` supports `language` and MiMo `audio_tag`.
- WER is comparable to official `MimoAudio.asr_sft()` on a small sample set.

## Risks And Mitigations

### Risk: SGLang Version Not Installed In Dev Environment

The current local environment may not have `sglang`, `torch`, or `transformers` installed even though they are pinned in `pyproject.toml`.

Mitigation:

- Implement code structurally first.
- Validate import/load behavior in the real runtime environment or container.

### Risk: MiMo Text Backbone Is Not Directly Qwen2-Compatible

Phase 0 confirmed Qwen2 compatibility (`model_type: qwen2`, official `Qwen2Model` backbone). Residual risk is SGLang fusion / quant edge cases, not architecture mismatch.

Mitigation:

- Use SGLang `Qwen2ForCausalLM` for `model.*`.
- Log unmatched keys on first load; compare smoke output to official `asr_sft()`.

### Risk: Audio Tokenizer Requires Extra Dependencies

Mitigation:

- Isolate audio tokenizer loading in an adapter.
- Fail with a clear error if `audio_tokenizer_path` or dependencies are missing.
- Keep tokenizer path configurable.

### Risk: Audio Code Embedding Path Is More Complex Than Qwen3-ASR

Mitigation:

- Implement single-request path first.
- Disable CUDA graph initially.
- Add batching only after placeholder/code alignment is verified.

### Risk: Weight Loading Mismatch

Mitigation:

- Implement explicit prefix mapping.
- Log unmatched required keys during development.
- Skip only prefixes that are **absent** from the ASR checkpoint (code2wav/vocoder).
- Do **not** skip `local_transformer*` — required for decode.
- Compare generated text with official `MimoAudio.asr_sft()` on the same audio.

### Risk: Dual-Channel Decode Hard To Map Onto SGLang Scheduler

MiMo ASR requires `slm_sample`-style decode with `local_forward` on `<\|empty\|>` steps. Default `OmniScheduler` + `ModelRunner._sample_next_token_ids` assumes **one text token per forward** (see `qwen3_asr/stages.py` vs `moss_tts_local/model_runner.py`).

Mitigation:

- Treat Phase 5 as a **first-class workstream**, not an implementation detail of `sglang_model.py`.
- Start with Option C (single-request custom runner); document `max_new_tokens` unit mapping.
- Port official prefill/decode; use vLLM-Omni as second reference.
- Disable CUDA graph until parity tests pass.
- Compare transcript (and optionally first decode step) to official `MimoAudio.asr_sft()`.

### Risk: Prompt Mismatch Causes Bad ASR

Mitigation:

- Copy official `asr_sft()` prompt behavior exactly.
- Add a debug mode to dump prompt text and token ids.
- Compare tokenized prompts against official implementation.

## Open Questions (Closed)

All items below were resolved in Phase 0. See that section for details.

| # | Question | Answer |
|---|----------|--------|
| 1 | HF `architectures` value? | `MiMoV2ASRForCausalLM` |
| 2 | Text config field name? | None — flat Qwen2 top-level config |
| 3 | Special token ids? | `<\|empty\|>`=151667; stop=151645; see Phase 0 table |
| 4 | Tokenizer output shape? | `[T, 8]` int RVQ codes |
| 5 | `<\|empty\|>` per code frame/group? | **1 empty token per 4 code frames** (`group_size`) |
| 6 | ASR vs skip prefixes? | Load all checkpoint prefixes; no code2wav in ASR ckpt; **`local_transformer*` required** |
| 7 | Reuse SGLang Qwen2? | **Yes** — `Qwen2ForCausalLM` for `model.*` |
| 8 | MRoPE? | **No** — ordinary RoPE |
| 9 | Stop tokens? | `151645` (`<\|redacted_im_end\|>`) |
| 10 | Greedy or sampling? | **Text greedy**; **local sampling** on empty steps |
