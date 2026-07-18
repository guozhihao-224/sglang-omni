#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Probe Equal-KV MAX_TOTAL_TOKENS for same-GPU DP + WEIGHT_IPC.
#
# For each T in TOKENS_LIST: launch → record pass/fail + nvidia-smi → tear down.
# Writes a Markdown + CSV summary under OUT_DIR for PR notes.
#
# Example (H100 NUMA0 / GPU2):
#   export PATH=/path/to/.venv/bin:$PATH
#   GPU_ID=2 N=4 CORE_BLOCKS="0-7 8-15 16-23 24-31" \
#     TOKENS_LIST="100000 110000 120000 130000" \
#     bash examples/weight_ipc/probe_equal_kv.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCH="$ROOT/examples/mps_dp/launch.sh"

GPU_ID=${GPU_ID:-0}
N=${N:-4}
CORE_BLOCKS=${CORE_BLOCKS:-}
MODEL=${MODEL:-bosonai/higgs-tts-3-4b}
BASE_PORT=${BASE_PORT:-8801}
WEIGHT_IPC=${WEIGHT_IPC:-1}
TOKENS_LIST=${TOKENS_LIST:-"100000 110000 120000 130000"}
OUT_DIR=${OUT_DIR:-$ROOT/results/weight_ipc_kv_probe_$(date +%Y%m%d-%H%M%S)}

die() { echo "error: $*" >&2; exit 1; }

command -v sgl-omni >/dev/null || die "sgl-omni not on PATH"
command -v numactl >/dev/null || die "numactl is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
[ -n "$CORE_BLOCKS" ] || die "CORE_BLOCKS is required"

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/probe.csv"
MD="$OUT_DIR/SUMMARY.md"
LOG="$OUT_DIR/probe.log"

echo "tokens,result,resolved_tokens,gpu_mem_used_mib,gpu_mem_total_mib,state_dir,notes" > "$CSV"
{
  echo "# Equal-KV probe (WEIGHT_IPC=$WEIGHT_IPC)"
  echo
  echo "| Field | Value |"
  echo "|---|---|"
  echo "| Date | $(date -u +%Y-%m-%dT%H:%M:%SZ) |"
  echo "| Host | $(hostname) |"
  echo "| GPU | $GPU_ID ($(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader)) |"
  echo "| N | $N |"
  echo "| WEIGHT_IPC | $WEIGHT_IPC |"
  echo "| MODEL | $MODEL |"
  echo "| CORE_BLOCKS | \`$CORE_BLOCKS\` |"
  echo "| TOKENS_LIST | $TOKENS_LIST |"
  echo
  echo "## Results"
  echo
  echo "| T | Result | Resolved #tokens | GPU mem (MiB) | Notes |"
  echo "|---:|---|---|---:|---|"
} > "$MD"

latest_run_for_gpu() {
  local gpu=$1
  local state_root=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
  ls -1dt "$state_root"/gpu-"$gpu"/run-* 2>/dev/null | head -n1
}

gpu_mem_used() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

gpu_mem_total() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' '
}

# Best-effort: tear down any leftover run on this GPU before probing.
pre_clean() {
  local st
  st=$(latest_run_for_gpu "$GPU_ID" || true)
  if [ -n "${st:-}" ] && [ -d "$st" ]; then
    echo "==> pre-clean: down $(basename "$st")" | tee -a "$LOG"
    bash "$LAUNCH" down "$(basename "$st")" >>"$LOG" 2>&1 || true
  fi
}

probe_one() {
  local tokens=$1
  local attempt_log="$OUT_DIR/T${tokens}.log"
  local result=FAIL
  local resolved="-"
  local notes=""
  local state=""
  local mem_used mem_total

  echo "" | tee -a "$LOG"
  echo "==> probe T=$tokens WEIGHT_IPC=$WEIGHT_IPC N=$N GPU=$GPU_ID" | tee -a "$LOG"

  set +e
  WEIGHT_IPC=$WEIGHT_IPC N=$N GPU_ID=$GPU_ID CORE_BLOCKS="$CORE_BLOCKS" \
    MAX_TOTAL_TOKENS=$tokens MODEL="$MODEL" BASE_PORT=$BASE_PORT \
    bash "$LAUNCH" up >"$attempt_log" 2>&1
  local rc=$?
  set -e

  cat "$attempt_log" >> "$LOG"
  state=$(latest_run_for_gpu "$GPU_ID" || true)

  mem_total=$(gpu_mem_total)
  # Snapshot while servers are up (PASS) or just after failed launch (FAIL).
  mem_used=$(gpu_mem_used)

  if [ "$rc" -eq 0 ]; then
    result=PASS
    if [ -n "$state" ]; then
      # Collect resolved tokens from all replicas (must match).
      resolved=$(grep -hE 'KV #tokens:[[:space:]]*[0-9]+' "$attempt_log" \
        | grep -oE '[0-9]+$' | sort -u | tr '\n' '/' | sed 's|/$||')
      [ -n "$resolved" ] || resolved="$tokens"
      notes="state=$(basename "$state")"
      nvidia-smi -i "$GPU_ID" > "$OUT_DIR/T${tokens}_nvidia-smi.txt"
      nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,used_gpu_memory,process_name \
        --format=csv > "$OUT_DIR/T${tokens}_procs.csv" || true
    else
      notes="pass but missing state dir"
    fi
  else
    result=FAIL
    if grep -qE 'resolved .* KV tokens; expected|CUDA out of memory|OutOfMemory' "$attempt_log"; then
      notes=$(grep -m1 -E 'resolved .* KV tokens; expected|CUDA out of memory|OutOfMemory' "$attempt_log" \
        | tr ',' ';' | head -c 200)
    else
      notes="see T${tokens}.log"
    fi
  fi

  if [ -n "${state:-}" ]; then
    echo "==> tearing down $(basename "$state") ($result)" | tee -a "$LOG"
    bash "$LAUNCH" down "$(basename "$state")" >>"$LOG" 2>&1 || true
  fi
  # Drain GPU before next probe (PASS and FAIL).
  local i=0
  while [ "$(gpu_mem_used)" -gt 512 ] && [ "$i" -lt 60 ]; do
    sleep 2
    i=$((i + 1))
  done

  echo "$tokens,$result,$resolved,$mem_used,$mem_total,${state:-},$notes" >> "$CSV"
  echo "| $tokens | **$result** | $resolved | ${mem_used}/${mem_total} | $notes |" >> "$MD"
  echo "==> T=$tokens → $result (mem ${mem_used}/${mem_total} MiB) $notes" | tee -a "$LOG"
}

pre_clean

for t in $TOKENS_LIST; do
  probe_one "$t"
done

{
  echo
  echo "## Artifacts"
  echo
  echo "- CSV: \`$CSV\`"
  echo "- Full log: \`$LOG\`"
  echo "- Per-T launch logs: \`$OUT_DIR/T*.log\`"
  echo
  echo "## Interpretation"
  echo
  echo "Largest PASS under Equal-KV + WEIGHT_IPC is the practical upper bound for this host/model/N."
  echo "Raise production \`MAX_TOTAL_TOKENS\` only after a PASS with headroom for graph/runtime peaks."
} >> "$MD"

echo ""
echo "DONE. Summary: $MD"
cat "$MD"
