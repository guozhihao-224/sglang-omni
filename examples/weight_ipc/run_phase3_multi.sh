#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run Phase-3 weight-IPC perf matrix multiple times at a fixed Equal KV.
#
# Example:
#   export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=$PWD
#   GPU_ID=0 MAX_TOTAL_TOKENS=100000 TRIALS=3 \
#   CORE_BLOCKS_DP3="0-7 8-15 16-23" \
#   CORE_BLOCKS_DP4="0-7 8-15 16-23 24-31" \
#     bash examples/weight_ipc/run_phase3_multi.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PHASE3="$ROOT/examples/weight_ipc/run_phase3_perf.sh"
LAUNCH="$ROOT/examples/mps_dp/launch.sh"
AGG_MULTI="$ROOT/examples/weight_ipc/aggregate_phase3_trials.py"

TRIALS=${TRIALS:-3}
GPU_ID=${GPU_ID:-0}
TOKENS=${MAX_TOTAL_TOKENS:-100000}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/weight_ipc_phase3_multi_${TOKENS}_$(date +%Y%m%d-%H%M%S)}
SAMPLES=${SAMPLES:-100}
TTFC_SAMPLES=${TTFC_SAMPLES:-150}
CONCURRENCY=${CONCURRENCY:-64}
TTFC_OFFERED_QPS=${TTFC_OFFERED_QPS:-34}

die() { echo "error: $*" >&2; exit 1; }

[ -n "${CORE_BLOCKS_DP3:-}" ] || die "CORE_BLOCKS_DP3 required"
[ -n "${CORE_BLOCKS_DP4:-}" ] || die "CORE_BLOCKS_DP4 required"
command -v sgl-omni >/dev/null || die "sgl-omni not on PATH"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/multi.log"
echo "OUT_ROOT=$OUT_ROOT TRIALS=$TRIALS MAX_TOTAL_TOKENS=$TOKENS" | tee "$LOG"

down_gpu() {
  local state_root=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
  local d
  for d in "$state_root"/gpu-"$GPU_ID"/run-*; do
    [ -d "$d" ] || continue
    echo "==> down $(basename "$d")" | tee -a "$LOG"
    bash "$LAUNCH" down "$(basename "$d")" >>"$LOG" 2>&1 || true
  done
  local i=0 mem
  while true; do
    mem=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$mem" -le 512 ] && break
    i=$((i + 1))
    [ "$i" -ge 60 ] && break
    sleep 2
  done
}

trial_ok=0
trial_fail=0
for t in $(seq 1 "$TRIALS"); do
  echo "" | tee -a "$LOG"
  echo "======== TRIAL $t / $TRIALS ========" | tee -a "$LOG"
  down_gpu
  trial_dir="$OUT_ROOT/trial_$t"
  set +e
  GPU_ID=$GPU_ID MAX_TOTAL_TOKENS=$TOKENS \
    CORE_BLOCKS_DP3="$CORE_BLOCKS_DP3" CORE_BLOCKS_DP4="$CORE_BLOCKS_DP4" \
    SAMPLES=$SAMPLES TTFC_SAMPLES=$TTFC_SAMPLES CONCURRENCY=$CONCURRENCY \
    TTFC_OFFERED_QPS=$TTFC_OFFERED_QPS \
    OUT_ROOT="$trial_dir" \
    bash "$PHASE3" >>"$LOG" 2>&1
  rc=$?
  set -e
  down_gpu
  if [ "$rc" -eq 0 ] && [ -f "$trial_dir/phase3_summary.json" ]; then
    trial_ok=$((trial_ok + 1))
    echo "==> trial $t OK" | tee -a "$LOG"
  else
    trial_fail=$((trial_fail + 1))
    echo "==> trial $t FAILED rc=$rc (see $LOG)" | tee -a "$LOG"
  fi
done

python "$AGG_MULTI" "$OUT_ROOT" --out "$OUT_ROOT/multi_summary.json" --md "$OUT_ROOT/SUMMARY.md"
echo ""
echo "DONE trials_ok=$trial_ok trials_fail=$trial_fail"
echo "Summary: $OUT_ROOT/SUMMARY.md"
cat "$OUT_ROOT/SUMMARY.md"
[ "$trial_ok" -ge 1 ] || die "no successful trials"
