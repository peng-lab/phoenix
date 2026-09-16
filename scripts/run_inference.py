#!/usr/bin/env python
"""
Run Phoenix inference over a list of SpatialData `.zarr` stores on a single device.

Each store is predicted and gets a `pred_table` written back into it, so ground truth and prediction
live side by side in one object for later plotting. Prediction only -- no metrics, no figures.

This script deliberately knows about exactly one device and takes an explicit list of stores; it does
not glob a directory and it does not fan out across GPUs itself. Splitting work across the GPUs of a
node and keeping all of them busy is a SLURM-and-bash concern, handled by `inference_node.sh`, not a
concern of this file -- see `run_inference.sbatch`.

Stores that already hold a `pred_table` are skipped unless `--overwrite` is given, which makes a run
resumable: if the job hits its wall clock, resubmitting it picks up where it stopped.

Usage
-----
    python run_inference.py \
        --weights /path/to/flow_model.pth \
        --panel /path/to/xenium_human_multi.npy \
        --stats /path/to/stats_table.npz \
        store_a.zarr store_b.zarr store_c.zarr

© Peng Lab / Helmholtz Munich
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phoenix_inference import load_model, predict_slide

logger = logging.getLogger("run_inference")

# spatialdata's ome-zarr reader logs an INFO line per root attribute and per multiscale level, which
# buries this script's own progress once a store is opened per slide. Quietened per logger rather than
# by raising the root level, so this script's own INFO output survives.
NOISY_LOGGERS = ("ome_zarr", "spatialdata")

# -------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv
        Argument list; defaults to `sys.argv[1:]`.

    Returns
    -------
    The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict spatial gene expression for a list of SpatialData zarr stores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("stores", type=Path, nargs="+", help="SpatialData .zarr stores to predict")

    parser.add_argument("--weights", type=Path, required=True, help="flow_model.pth checkpoint")
    parser.add_argument("--panel", type=Path, required=True, help="gene panel .npy, in panel order")
    parser.add_argument("--stats", type=Path, required=True, help="stats_table.npz with 'mean' and 'std'")

    parser.add_argument("--table-key", default="table", help="ground-truth table to read")
    parser.add_argument("--pred-key", default="pred_table", help="table the prediction is written to")

    parser.add_argument("--batch-size", type=int, default=128, help="cells per forward pass")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per slide")
    parser.add_argument("--patch-size", type=int, default=224, help="patch side length at --target-mpp")
    parser.add_argument("--target-mpp", type=float, default=0.5, help="target microns per pixel")

    parser.add_argument("--atol", type=float, default=1e-1, help="ODE solver absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-1, help="ODE solver relative tolerance")
    parser.add_argument("--seed", type=int, default=0, help="seed, applied per slide")

    parser.add_argument(
        "--device",
        default=None,
        help="torch device; default is 'cuda' if visible, else 'cpu'",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the K/V-caching sampler (CUDA and Ampere+ only; slightly different numerics)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --pred-key")

    return parser.parse_args(argv)


def configure_logging(prefix: str) -> None:
    """
    Send this process's log to stdout, tagged with who is speaking.

    Parameters
    ----------
    prefix
        Tag for the log line, e.g. the device this process runs on.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{prefix}] %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def resolve_device(spec: str | None) -> str:
    """
    Work out which device to run on.

    Parameters
    ----------
    spec
        Explicit torch device, or `None` to autodetect.

    Returns
    -------
    The device string this process runs on. Under `inference_node.sh`, `CUDA_VISIBLE_DEVICES` already
    restricts this process to one GPU, so "cuda" (with no index) is always the right default there.
    """
    if spec:
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    """
    Predict every given store on one device and report what happened.

    Parameters
    ----------
    argv
        Argument list; defaults to `sys.argv[1:]`.

    Returns
    -------
    Process exit status: non-zero if any slide failed.
    """
    args = parse_args(argv)
    device = resolve_device(args.device)
    configure_logging(device)

    for label, path in (("weights", args.weights), ("panel", args.panel), ("stats", args.stats)):
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            return 2

    logger.info(
        "%d store(s) on %s; seed=%d atol=%g rtol=%g batch_size=%d num_workers=%d fast=%s",
        len(args.stores),
        device,
        args.seed,
        args.atol,
        args.rtol,
        args.batch_size,
        args.num_workers,
        args.fast,
    )

    logger.info("loading %s", args.weights)
    model = load_model(args.weights, device)
    gene_list = list(np.load(args.panel))
    stats = np.load(args.stats)

    written, skipped, failed = [], [], []
    for path in args.stores:
        name = path.name
        start = time.perf_counter()
        try:
            n_cells = predict_slide(
                path,
                model,
                gene_list,
                stats,
                table_key=args.table_key,
                pred_key=args.pred_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                patch_size=args.patch_size,
                target_mpp=args.target_mpp,
                atol=args.atol,
                rtol=args.rtol,
                fast=args.fast,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception:
            logger.exception("%s: failed", name)
            failed.append(name)
            continue

        if n_cells is None:
            skipped.append(name)
            continue

        elapsed = time.perf_counter() - start
        logger.info("%s: done in %.1fs (%.1f cells/s)", name, elapsed, n_cells / elapsed)
        written.append(name)

    logger.info("%d written, %d skipped, %d failed", len(written), len(skipped), len(failed))
    for name in failed:
        logger.error("  failed %s", name)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
