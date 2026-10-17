#!/usr/bin/env bash
# Phoenix inference (run_inference_flash.py) with every SpatialData store split across all 4 GPUs of the node
# (Lightning DDP).
#
# Where run_inference_flash.sbatch gives each GPU its own store, this script runs the stores one after
# another and shards each one over the 4 GPUs: one SLURM task per GPU, each predicting every 4th cell,
# with rank 0 gathering the rows in table order and writing the same `pred_table` a single-GPU run writes.
# Use it for a few very large slides (a store that needs more than the 24 h wall clock on one GPU); for
# many small slides run_inference_flash.sbatch is the better choice, as it has no cross-rank waiting.
# Measured on a 56,510-cell store: 1 GPU 7191 s, 4 GPUs 1838 s (3.9x).
#
# Paths are taken from the environment so nothing site-specific is baked in:
#
#   sbatch --export=ALL,SDATA_DIR=/path/to/zarrs,WEIGHTS=/path/to/flow_model.pth,\
# PANEL=/path/to/xenium_human_multi.npy,STATS=/path/to/stats_table.npz run_inference_flash_ddp.sbatch
#
# Compute nodes have no internet, so all four must already exist on a shared filesystem.
#
# REPO and VENV default to the main checkout; override them (same --export mechanism) to run against a
# worktree, e.g. --export=ALL,REPO=/path/to/worktree,... The code under $REPO/src is what runs.
#
# Optional: FAST=1 for the K/V-caching sampler.
#
# Stores that already hold a pred_table are skipped, so if the 24 h wall clock (the QoS maximum) is not
# enough for every slide, just resubmit the same job. Each store is its own srun step: if one fails, its
# ranks are taken down together, the store is logged in failed.txt, and the job moves on to the next.
#
# © Peng Lab / Helmholtz Munich

#SBATCH --job-name=phoenix-ddp-inference
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

# Tunables; override on the sbatch --export line. BATCH_SIZE is per GPU.
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
ATOL=${ATOL:-0.1}
RTOL=${RTOL:-0.1}
SEED=${SEED:-0}

# 16 cores per task, 4 tasks per node: 8 loader workers each leaves headroom without each process's own
# BLAS calls (patch resize, de-normalisation) oversubscribing the rest.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

source "$VENV/bin/activate"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

LOG_DIR="$REPO/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

mapfile -t STORES < <(du -sb "$SDATA_DIR"/*.zarr | sort -rn | cut -f2)
if [[ ${#STORES[@]} -eq 0 ]]; then
    echo "no .zarr stores found in $SDATA_DIR" >&2
    exit 2
fi
echo "${#STORES[@]} store(s), one after another, each split across 4 gpu(s)"

# One task per GPU; Lightning reads rank and world size from the SLURM environment. Every task must see all
# four GPUs, because Lightning selects cuda:$SLURM_LOCALID: hence --gpu-bind=none (SLURM otherwise binds one
# GPU per task) and no --gpus-per-task. `--cpu-bind=none` and `--gres` are repeated on the srun line because
# task/cgroup and task/affinity are both active on dc-hwai: a step that does not re-request them can otherwise
# see fewer GPUs, or a narrower core mask, than the job holds. --kill-on-bad-exit takes the whole step down
# when one rank dies instead of leaving the others hanging in a collective. -u (--unbuffered) is needed for the
# progress bar: tqdm redraws with '\r' and only ends a line when the bar closes, and srun holds back output
# until a newline, so without it the log shows nothing until the whole store is done.
for store in "${STORES[@]}"; do
    srun -u --ntasks-per-node=4 --gres=gpu:4 --gpu-bind=none --cpu-bind=none --kill-on-bad-exit=1 \
        python "$REPO/scripts/run_inference_flash.py" \
        --weights "$WEIGHTS" --panel "$PANEL" --stats "$STATS" --devices 4 \
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --atol "$ATOL" --rtol "$RTOL" --seed "$SEED" ${FAST:+--fast} \
        "$store" > "$LOG_DIR/$(basename "$store" .zarr).log" 2>&1 \
        || echo "$store" >> "$LOG_DIR/failed.txt"
done

if [[ -s "$LOG_DIR/failed.txt" ]]; then
    echo "$(wc -l < "$LOG_DIR/failed.txt") of ${#STORES[@]} store(s) failed:" >&2
    cat "$LOG_DIR/failed.txt" >&2
    exit 1
fi
echo "all ${#STORES[@]} store(s) done"
