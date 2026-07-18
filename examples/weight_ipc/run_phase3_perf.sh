#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Phase 3 weight-IPC performance: DP3 unshared vs shared, then DP4 shared.
#
# Uses one SeedTTS generate-only client per replica (same discipline as mps_dp
# case study). Requires sgl-omni + PYTHONPATH to the repo root.
#
# Example:
#   export PATH=$PWD/.venv/bin:$PATH
#   export PYTHONPATH=$PWD
#   GPU_ID=0 \
#   CORE_BLOCKS_DP3="0-7 8-15 16-23" \
#   CORE_BLOCKS_DP4="0-7 8-15 16-23 24-31" \
#     bash examples/weight_ipc/run_phase3_perf.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCH="$ROOT/examples/mps_dp/launch.sh"
AGG="$ROOT/examples/weight_ipc/aggregate_speed_results.py"
OUT_ROOT=${OUT_ROOT:-$ROOT/results/weight_ipc_phase3_$(date +%Y%m%d-%H%M%S)}
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-bosonai/higgs-tts-3-4b}
MODEL_NAME=${MODEL_NAME:-higgs}
TOKENS=${MAX_TOTAL_TOKENS:-100000}
BASE_PORT=${BASE_PORT:-8801}
SAMPLES=${SAMPLES:-100}
CONCURRENCY=${CONCURRENCY:-64}
META=${META:-zhaochenyang20/seed-tts-eval-arrow}
WARMUP=${WARMUP:-2}
# Fixed offered load for TTFC comparison (aggregate across replicas).
TTFC_OFFERED_QPS=${TTFC_OFFERED_QPS:-34}
TTFC_SAMPLES=${TTFC_SAMPLES:-200}

die() { echo "error: $*" >&2; exit 1; }

need_tools() {
  command -v sgl-omni >/dev/null || die "sgl-omni not on PATH"
  command -v numactl >/dev/null || die "numactl required"
  [ -n "${PYTHONPATH:-}" ] || export PYTHONPATH="$ROOT"
}

latest_run_for_gpu() {
  local gpu=$1
  local state_root=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
  local matches
  # Avoid set -e / pipefail failure when the glob matches nothing.
  matches=$(find "$state_root/gpu-$gpu" -maxdepth 1 -type d -name 'run-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-)
  printf '%s' "$matches"
}

down_latest() {
  local state run
  state=$(latest_run_for_gpu "$GPU_ID" || true)
  [ -n "$state" ] || { echo "==> no active run for GPU $GPU_ID"; return 0; }
  run=$(basename "$state")
  echo "==> tearing down $run"
  bash "$LAUNCH" down "$run" || true
  sleep 2
}

launch_pool() {
  local n=$1 weight_ipc=$2 blocks=$3 label=$4
  [ -n "$blocks" ] || die "core blocks empty for $label"
  echo "==> launch $label N=$n WEIGHT_IPC=$weight_ipc tokens=$TOKENS"
  WEIGHT_IPC=$weight_ipc N=$n GPU_ID=$GPU_ID CORE_BLOCKS="$blocks" \
    MAX_TOTAL_TOKENS=$TOKENS MODEL="$MODEL" MODEL_NAME="$MODEL_NAME" \
    BASE_PORT=$BASE_PORT \
    bash "$LAUNCH" up
  local state
  state=$(latest_run_for_gpu "$GPU_ID")
  [ -n "$state" ] || die "no state after launch $label"
  echo "$state" > "$OUT_ROOT/$label.state"
  echo "$state"
}

run_clients() {
  local n=$1 label=$2 stream_flag=$3 rate=$4 samples=$5
  local out_base=$OUT_ROOT/$label
  mkdir -p "$out_base"
  local pids=() i port out
  for ((i=0; i<n; i++)); do
    port=$((BASE_PORT + i))
    out=$out_base/replica_$i
    mkdir -p "$out"
    echo "==> client replica $i -> :$port  conc=$CONCURRENCY samples=$samples rate=$rate stream=$stream_flag"
    (
      cd "$ROOT"
      PYTHONPATH="$ROOT" python -m benchmarks.eval.benchmark_tts_seedtts \
        --use-existing-server \
        --generate-only \
        --base-url "http://127.0.0.1:$port" \
        --model "$MODEL_NAME" \
        --meta "$META" \
        --ref-format references \
        --max-samples "$samples" \
        --concurrency "$CONCURRENCY" \
        --request-rate "$rate" \
        --warmup "$WARMUP" \
        --output-dir "$out" \
        --disable-tqdm \
        $stream_flag
    ) > "$out/client.log" 2>&1 &
    pids+=($!)
  done
  local pid rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  [ "$rc" = 0 ] || {
    echo "error: one or more clients failed for $label; see $out_base/*/client.log" >&2
    return 1
  }
  local files=()
  for ((i=0; i<n; i++)); do
    files+=("$out_base/replica_$i/speed_results.json")
    [ -f "${files[-1]}" ] || die "missing ${files[-1]}"
  done
  python "$AGG" "${files[@]}" --label "$label" --out "$out_base/aggregate.json"
}

compare_dp3() {
  local u=$OUT_ROOT/dp3_unshared_peak/aggregate.json
  local s=$OUT_ROOT/dp3_shared_peak/aggregate.json
  python - <<PY
import json
from pathlib import Path
u = json.loads(Path("$u").read_text())
s = json.loads(Path("$s").read_text())
uq, sq = u["aggregate_throughput_qps"], s["aggregate_throughput_qps"]
delta = (sq - uq) / uq * 100 if uq else float("nan")
# G2: sharing must not cost more than ~2% throughput (faster is fine).
out = {
  "dp3_unshared_qps": uq,
  "dp3_shared_qps": sq,
  "shared_vs_unshared_pct": round(delta, 3),
  "regression_pct": round(min(0.0, delta), 3),
  "within_2pct_abs": abs(delta) <= 2.0,
  "g2_pass": delta >= -2.0,
}
Path("$OUT_ROOT/dp3_compare.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
if not out["g2_pass"]:
    print("WARNING: G2 throughput gate not met (|delta|>2%); continuing Phase 3", flush=True)
PY
}

main() {
  need_tools
  mkdir -p "$OUT_ROOT"
  echo "OUT_ROOT=$OUT_ROOT"
  local blocks3=${CORE_BLOCKS_DP3:-${CORE_BLOCKS:-}}
  local blocks4=${CORE_BLOCKS_DP4:-}
  [ -n "$blocks3" ] || die "set CORE_BLOCKS_DP3 (3 blocks)"
  [ -n "$blocks4" ] || die "set CORE_BLOCKS_DP4 (4 blocks)"

  # --- DP3 unshared peak ---
  down_latest
  launch_pool 3 0 "$blocks3" dp3_unshared_peak
  run_clients 3 dp3_unshared_peak "--stream" inf "$SAMPLES"
  down_latest

  # --- DP3 shared peak ---
  launch_pool 3 1 "$blocks3" dp3_shared_peak
  run_clients 3 dp3_shared_peak "--stream" inf "$SAMPLES"
  compare_dp3

  # --- DP3 shared fixed offered-load TTFC ---
  local per_rate
  per_rate=$(python - <<PY
print(round(float("$TTFC_OFFERED_QPS") / 3.0, 3))
PY
)
  run_clients 3 dp3_shared_ttfc34 "--stream" "$per_rate" "$TTFC_SAMPLES"
  down_latest

  # --- DP4 shared peak + TTFC ---
  launch_pool 4 1 "$blocks4" dp4_shared_peak
  run_clients 4 dp4_shared_peak "--stream" inf "$SAMPLES"
  per_rate=$(python - <<PY
print(round(float("$TTFC_OFFERED_QPS") / 4.0, 3))
PY
)
  run_clients 4 dp4_shared_ttfc34 "--stream" "$per_rate" "$TTFC_SAMPLES"

  python - <<PY
import json
from pathlib import Path
root = Path("$OUT_ROOT")
summary = {
  "out_root": str(root),
  "dp3_compare": json.loads((root / "dp3_compare.json").read_text()),
  "dp3_shared_ttfc34": json.loads((root / "dp3_shared_ttfc34/aggregate.json").read_text()),
  "dp4_shared_peak": json.loads((root / "dp4_shared_peak/aggregate.json").read_text()),
  "dp4_shared_ttfc34": json.loads((root / "dp4_shared_ttfc34/aggregate.json").read_text()),
}
(root / "phase3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

  echo "==> Phase 3 artifacts under $OUT_ROOT"
  local dp4_run
  dp4_run=$(basename "$(cat "$OUT_ROOT/dp4_shared_peak.state")")
  echo "==> servers still up (dp4); tear down with: bash $LAUNCH down $dp4_run"
}

main "$@"
