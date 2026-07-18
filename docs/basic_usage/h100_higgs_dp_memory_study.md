# H100 Higgs same-GPU DP memory study (Phase A/B draft)

> Tracking: [sgl-project/sglang-omni#1052](https://github.com/sgl-project/sglang-omni/issues/1052) · Task 3 (CUDA IPC memory study)  
> Goal: quantify per-replica VRAM and decide whether replicated weights are the DP3→DP4 wall **before** implementing cross-process weight sharing.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (GPU 2) |
| Model | `bosonai/higgs-tts-3-4b` (snapshot `7556c17e…`) |
| Launcher | `examples/mps_dp/launch.sh` |
| Equal KV | `--max-total-tokens` + launcher checks `#tokens` |
| MPS | private daemon; attach verified on successful runs |
| CPU | NUMA node 0 physical cores `0–31` (no SMT siblings in `CORE_BLOCKS`) |

## Feasibility matrix

| Config | Equal KV T | Result |
|---|---:|---|
| DP1 | profiled **426944** | ✓ |
| DP3 | **100000** | ✓ (recipe) |
| DP4 | 50000 | ✓ |
| DP4 | **70000** | ✓ |
| DP4 | 80000 | ✗ (replica3 resolved **52449**) |
| DP4 | 100000 | ✗ (replica3 resolved **2836**) |

**DP4 Equal-KV upper bound: `T_max ∈ [70000, 80000)`** (finest probe not required for the Go/No-Go).

## Per-replica breakdown (from startup logs)

AR `tts_engine` path, consistent across replicas:

| Component | Size | Notes |
|---|---:|---|
| Weights | **7.60 GB** | `Load weight … mem usage=7.60 GB` |
| KV @ 100000 | **13.74 GB** | K 6.87 + V 6.87 |
| KV per token | **~137 KB** | 13.74 GB / 1e5 |
| CUDA Graph pool | **~0.29 GB** | capture `bs=[1..64]` |
| Other (approx.) | residual | encoder/codec/runtime + allocator slack; not fully isolated yet |

**Implied fixed cost per replica (ex-KV):** ≈ **7.9 GB** (weights + graph), plus non-AR / other runtime not fully split in this pass.

## DP4 @ 100000 failure trace (sequential start)

| Replica | avail before weight load | Weights | Resolved `#tokens` |
|---|---:|---:|---:|
| 0 | 77.94 GB | 7.60 GB | 100000 |
| 1 | 55.09 GB | 7.60 GB | 100000 |
| 2 | 32.25 GB | 7.60 GB | 100000 |
| 3 | **9.41 GB** | 7.60 GB | **2836** |

Replica 3 still loads a full weight copy; remaining free memory cannot fund Equal KV=100k (nor 80k).

## Attribution: is replicated weight the DP3→DP4 wall?

**Short answer: not by itself.**

1. **DP4 is feasible** at Equal KV=70k → four weight copies fit on the card.
2. **DP4 fails at Equal KV=100k/80k** because after 3×(weights+KV+graph), replica 4 lacks free memory for the **same** KV cap — the binding constraint at the recipe target is **KV headroom under N× replication**, not “weights alone make N=4 impossible.”
3. **Replicated weights are still a first-order cost**: each extra replica burns ~7.6 GB before any KV. That shrinks the per-replica KV budget and is why DP4 cannot match DP3’s 100k Equal KV.

**Working model**

```text
card ≈ N × (Weights 7.6 + Graph 0.3 + Other) + N × KV(T)
```

At N=4 and T=100k, KV alone is ~4×13.74 ≈ 55 GB; plus ~4×7.9 ≈ 32 GB fixed → exceeds ~80 GB usable.  
At N=4 and T=70k, the sum fits; at T=80k it does not under Equal-KV enforcement.

## CUDA IPC weight sharing — Go/No-Go (preliminary)

| Question | Answer |
|---|---|
| Does IPC unlock DP4 at all? | **No need** — DP4 already works at T≤70k. |
| Does IPC help the interesting target (DP4 @ ~100k Equal KV)? | **Likely yes (upper bound).** Saving ~(N−1)×7.6 ≈ **22.8 GB** at N=4 could fund ~22.8GB / 137KB ≈ **~160k tokens of aggregate KV**, i.e. enough headroom to push Equal T from ~70k toward **100k+**, *if* graph/other do not eat the savings and sharing is graph/MPS-safe. |
| Is weight IPC the highest-leverage next step? | **Conditional.** It is a KV-capacity unlock for higher-N Equal KV, not a fix for host-bound QPS. Prefer after confirming single-worker / graph tradeoffs (tasks 4–5). |

**Recommendation**

- **Do not implement production weight IPC yet.**
- Mark Phase B measurement **sufficient for a first conclusion**.
- Optional Phase C: 1–2 page feasibility note (allocator offset, CUDA Graph, MPS read-only share) + **benefit bound only**; PoC only if product wants DP4@≥100k Equal KV on H100 Higgs.

## One-line summary

> 在 H100 Higgs 上，DP4 跑得起来（Equal KV≈70k），DP3→DP4@100k 的墙是 **N 份 weights 挤压后的 KV 余量**，不是「权重复制导致根本装不下第四份」；CUDA IPC 共享权重最多省 ~23GB，**有望把 DP4 Equal KV 从 ~70k 抬到 100k 量级**，但是否值得做取决于要不要这个 KV 目标，以及单 worker / graph 优化是否更优先。

## What this pass did / did not measure

**Did:** startup-time weights / KV / graph sizes; DP1/DP3/DP4 Equal-KV feasibility; MPS attach on success paths.

**Did not yet:** end-to-end QPS; full “Other” isolation (encoder/codec); reserved vs allocated; graph off ablation; IPC PoC.

## Next (optional)

- [ ] One more probe at `MAX_TOTAL_TOKENS=75000` to tighten `T_max`
- [ ] Record `nvidia-smi` card-level used at DP3@100k and DP4@70k
- [ ] Phase C Go/No-Go memo (design-only) if DP4@100k is a hard product goal
