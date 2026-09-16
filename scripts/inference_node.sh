#!/usr/bin/env bash
# Per-node fan-out for Phoenix inference: keep every GPU on this node busy with a dynamic queue.
#
# Runs once per node under `srun --ntasks-per-node=1` (see run_inference.sbatch), which is what makes
# `SLURM_NODEID`/`SLURM_NNODES` meaningful here. Each node takes a static, round-robin slice of the
# shared store list (interleaved, not contiguous, because STORES_FILE is sorted largest-first and a
# contiguous split would hand every big slide to node 0), then predicts that slice with four
# single-GPU `run_inference.py` processes, refilling a GPU as soon as it frees up -- the dynamic half
# of "static across nodes, dynamic within each node".
#
# `run_inference.py` itself knows nothing about any of this: it is handed one GPU via
# `CUDA_VISIBLE_DEVICES` and one store, and predicts it.
#
# Required environment (exported by run_inference.sbatch):
#   STORES_FILE   newline-separated store paths, one per line, largest first
#   LOG_DIR       directory for one log file per slide
#   REPO          repo root, so this script finds run_inference.py regardless of $PWD
#   WEIGHTS PANEL STATS BATCH_SIZE NUM_WORKERS ATOL RTOL SEED
#
# Optional:
#   N_GPUS        GPUs on this node (default 4, the dc-hwai node size)
#   DRY_RUN       if set, replace the python call with a short sleep -- exercises the queueing logic
#                 with no GPU and no model, e.g.:
#                   DRY_RUN=1 N_GPUS=4 N_NODES=1 NODE_ID=0 STORES_FILE=stores.txt LOG_DIR=/tmp \
#                       bash inference_node.sh
#
# © Peng Lab / Helmholtz Munich

set -euo pipefail

NODE_ID=${SLURM_NODEID:-${NODE_ID:-0}}
N_NODES=${SLURM_NNODES:-${N_NODES:-1}}
N_GPUS=${N_GPUS:-4}

: "${STORES_FILE:?set STORES_FILE to the newline-separated list of stores}"
: "${LOG_DIR:?set LOG_DIR to the directory for per-slide logs}"

echo "[node $NODE_ID/$N_NODES] $N_GPUS gpu(s), reading $STORES_FILE"

# Round-robin, not `split -n l/$N_NODES`: STORES_FILE is sorted largest-first, so a contiguous split
# would give node 0 every big slide and node 1 every small one. Interleaving gives each node a mix.
mapfile -t SHARD < <(awk -v n="$N_NODES" -v k="$NODE_ID" 'NR % n == k' "$STORES_FILE")
echo "[node $NODE_ID/$N_NODES] ${#SHARD[@]} store(s) assigned"

declare -A BUSY=()    # pid -> gpu index
declare -A RUNNING=() # pid -> store path, for failure reporting
FREE=()
for ((gpu = N_GPUS - 1; gpu >= 0; gpu--)); do
    FREE+=("$gpu")
done
FAILED=()

# Block until one tracked child exits, hand its GPU back to the free pool and record a non-zero exit.
reap() {
    local rc=0 pid
    wait -n -p pid "${!BUSY[@]}" || rc=$?
    FREE+=("${BUSY[$pid]}")
    if ((rc != 0)); then
        FAILED+=("${RUNNING[$pid]} (exit $rc)")
    fi
    unset "BUSY[$pid]" "RUNNING[$pid]"
}

for store in "${SHARD[@]}"; do
    if ((${#BUSY[@]} == N_GPUS)); then
        reap
    fi

    gpu=${FREE[-1]}
    unset 'FREE[-1]'
    log="$LOG_DIR/$(basename "$store" .zarr).log"

    if [[ -n "${DRY_RUN:-}" ]]; then
        (echo "[gpu $gpu] $store"; sleep "$((RANDOM % 5 + 1))") > "$log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$gpu python "$REPO/scripts/run_inference.py" \
            --weights "$WEIGHTS" --panel "$PANEL" --stats "$STATS" \
            --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
            --atol "$ATOL" --rtol "$RTOL" --seed "$SEED" \
            "$store" > "$log" 2>&1 &
    fi

    BUSY[$!]=$gpu
    RUNNING[$!]=$store
done

while ((${#BUSY[@]} > 0)); do
    reap
done

echo "[node $NODE_ID/$N_NODES] ${#SHARD[@]} store(s) attempted, ${#FAILED[@]} failed"
for entry in "${FAILED[@]}"; do
    echo "[node $NODE_ID/$N_NODES]   failed $entry"
done

((${#FAILED[@]} == 0))
