#!/usr/bin/env bash
# Phoenix inference (run_inference.py) over a directory of extracted SpatialData .zarr stores, on one 4-GPU node.
#
# Each store runs on its own GPU, so four stores are predicted at the same time. As soon as one finishes,
# the next store (largest first) takes its GPU. For a few very large slides, use run_inference_flash_ddp.sbatch.
#
# Paths are taken from the environment so nothing site-specific is baked in:
#
#   sbatch --export=ALL,SDATA_DIR=/path/to/zarrs,WEIGHTS=/path/to/flow_model.pth,\
# PANEL=/path/to/xenium_human_multi.npy,STATS=/path/to/stats_table.npz run_inference.sbatch
#
# Compute nodes have no internet, so all four must already exist on a shared filesystem.
#
# REPO and VENV default to the main checkout; override them (same --export mechanism) to run against a
# worktree, e.g. --export=ALL,REPO=/path/to/worktree,... The code under $REPO/src is what runs.
#
# Stores that already hold a pred_table are skipped, so if the 24 h wall clock (the QoS maximum) is not
# enough for every slide, just resubmit the same job. A failed store is logged in failed.txt and skipped.
#
# © Peng Lab / Helmholtz Munich

#SBATCH --job-name=phoenix-inference
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500GB
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO=${REPO:-/p/project1/hai_1240/phoenix}
VENV=${VENV:-/p/project1/hai_1240/phoenix/.venv}

: "${SDATA_DIR:?set SDATA_DIR to the directory of extracted .zarr stores}"
: "${WEIGHTS:?set WEIGHTS to the flow_model.pth checkpoint}"
: "${PANEL:?set PANEL to the gene panel .npy}"
: "${STATS:?set STATS to the stats_table.npz}"

# Tunables; override on the sbatch --export line.
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
ATOL=${ATOL:-0.1}
RTOL=${RTOL:-0.1}
SEED=${SEED:-0}

# 16 cores per GPU process: 8 loader workers each leaves headroom without each process's own
# BLAS calls (patch resize, de-normalisation) oversubscribing the rest.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

source "$VENV/bin/activate"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

LOG_DIR="$REPO/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Largest first, so the biggest slide does not start last and run alone at the end. `du -sb` is a size
# proxy, not a cell count (it includes the H&E image and transcript points), but it is good enough.
mapfile -t STORES < <(du -sb "$SDATA_DIR"/*.zarr | sort -rn | cut -f2)
if [[ ${#STORES[@]} -eq 0 ]]; then
    echo "no .zarr stores found in $SDATA_DIR" >&2
    exit 2
fi
echo "${#STORES[@]} store(s), one per gpu, 4 at a time"

# One srun step per store, at most 4 running at once. Each step asks for 1 GPU, 16 cores and 120G;
# --exact gives it only that, so SLURM places it on a free GPU and sets CUDA_VISIBLE_DEVICES for it.
# --mem is needed: without it a step claims all of the job's memory and the four would run one by one.
# `wait -n` returns when any step ends, which frees a slot for the next store.
for store in "${STORES[@]}"; do
    while (($(jobs -rp | wc -l) >= 4)); do
        wait -n || true
    done
    srun -u --exact --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=120G \
        python "$REPO/scripts/run_inference.py" \
        --weights "$WEIGHTS" --panel "$PANEL" --stats "$STATS" \
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --atol "$ATOL" --rtol "$RTOL" --seed "$SEED" \
        "$store" > "$LOG_DIR/$(basename "$store" .zarr).log" 2>&1 \
        || echo "$store" >> "$LOG_DIR/failed.txt" &
done
wait

if [[ -s "$LOG_DIR/failed.txt" ]]; then
    echo "$(wc -l < "$LOG_DIR/failed.txt") of ${#STORES[@]} store(s) failed:" >&2
    cat "$LOG_DIR/failed.txt" >&2
    exit 1
fi
echo "all ${#STORES[@]} store(s) done"
