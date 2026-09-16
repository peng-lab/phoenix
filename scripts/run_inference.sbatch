#!/usr/bin/env bash
# Batch Phoenix inference over a directory of extracted SpatialData .zarr stores.
#
# Fan-out is static across nodes and dynamic within each node: this script discovers the stores,
# orders them largest-first and hands the whole list to one `srun` task per node via STORES_FILE.
# `inference_node.sh` takes that node's round-robin slice and keeps all four GPUs busy with single-GPU
# `run_inference.py` processes, refilling a GPU as soon as it frees up. `run_inference.py` itself knows
# nothing about any of this -- it takes one device and a list of stores.
#
# Paths are taken from the environment so nothing site-specific is baked in:
#
#   sbatch --export=ALL,SDATA_DIR=/path/to/zarrs,WEIGHTS=/path/to/flow_model.pth,\
# PANEL=/path/to/xenium_human_multi.npy,STATS=/path/to/stats_table.npz run_inference.sbatch
#
# Compute nodes have no internet, so all four must already exist on a shared filesystem.
#
# Runs on 1 node by default; add `--nodes=2` on the sbatch command line for 2 (a CLI flag overrides
# the `#SBATCH --nodes` below -- #SBATCH lines are read by sbatch itself, not by the shell, so they
# cannot pick up a $NODES environment variable the way the --export'd values above do).
#
# Stores that already hold a pred_table are skipped, so if the 24 h wall clock (the QoS maximum) is
# not enough for every slide, just resubmit the same job.
#
# © Peng Lab / Helmholtz Munich

#SBATCH --job-name=phoenix-inference
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500GB
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO=/p/project1/hai_1240/phoenix

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

# 64 cores allocated, 4 concurrent processes per node: 8 loader workers each leaves headroom without
# each process's own BLAS calls (patch resize, de-normalisation) oversubscribing the rest.
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
export OMP_NUM_THREADS MKL_NUM_THREADS

source "$REPO/.venv/bin/activate"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

LOG_DIR="$REPO/scripts/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

STORES_FILE="$LOG_DIR/stores.txt"
# Largest-first: fed to a per-node dynamic queue, this is greedy longest-processing-time-first
# scheduling. `du -sb` is a size proxy, not a cell count (it includes the H&E image and transcripts
# points), but it costs nothing and is good enough to order by.
du -sb "$SDATA_DIR"/*.zarr | sort -rn | cut -f2 > "$STORES_FILE"

n_stores=$(wc -l < "$STORES_FILE")
echo "$n_stores store(s) across $SLURM_NNODES node(s), 4 gpu(s) each"

export STORES_FILE LOG_DIR REPO WEIGHTS PANEL STATS BATCH_SIZE NUM_WORKERS ATOL RTOL SEED

# One task per node (--ntasks-per-node=1 in the header); inference_node.sh does the 4-GPU fan-out for
# its node in bash. `--gres=gpu:4` and `--cpu-bind=none` are repeated on the srun line because
# task/cgroup and task/affinity are both active on dc-hwai: a step that does not re-request them can
# otherwise see fewer GPUs, or a narrower core mask, than the job actually holds.
srun --ntasks-per-node=1 --gres=gpu:4 --cpu-bind=none \
    bash "$REPO/scripts/inference_node.sh"
