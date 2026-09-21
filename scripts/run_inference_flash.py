#!/usr/bin/env python
"""
Run Phoenix inference over a list of SpatialData `.zarr` stores on a single device.

Identical to `run_inference.py` except for one line: the flow transformer is built from
`phoenix.models.flow_llama3` instead of `phoenix.models.flow_simple`, the optimized apex/flash-attn/
xformers reimplementation (see README.md's "Optimized model variant" section for the install steps).
The two are mathematically equivalent -- weights, image normalisation, gene panel order, de-
normalisation and the sampler are all unchanged -- so this script exists for throughput, not for a
different model.

`flow_llama3.FlashAttention.forward` hardcodes `@torch.autocast(device_type="cuda", ...)` and calls
`flash_attn_func`, so this script is CUDA-only (sm_80+); there is no CPU path, unlike
`run_inference.py`. Use `run_inference.py` on CPU or on GPUs older than Ampere.

Everything else -- the docstrings on `predict_slide`, resumability via `pred_table`, one store per
process, no multi-GPU fan-out here -- is exactly as documented in `run_inference.py`; see that file
for the fuller explanation. `inference_node.sh` / `run_inference.sbatch` still point at
`run_inference.py`; point them at this script instead to use the optimized kernels on a node with the
stack installed.

Usage
-----
    python run_inference_flash.py \
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
from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import spatialdata as sd
import timm
import torch
from spatialdata.models import TableModel
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2

from phoenix.datasets.zarr_dataset import SpatialDataset
from phoenix.helpers.inference import FlowPipeline
from phoenix.models.flow_llama3 import FlowTransformerConfig, FlowTransformerModel

logger = logging.getLogger("run_inference_flash")

# spatialdata's ome-zarr reader logs an INFO line per root attribute and per multiscale level, which
# buries this script's own progress once a store is opened per slide. Quietened per logger rather than
# by raising the root level, so this script's own INFO output survives.
NOISY_LOGGERS = ("ome_zarr", "spatialdata")

# Every real Xenium store's `obs` frame (the row index, `cell_id`, `instance_id`,
# `segmentation_method`, ...) uses pandas' nullable string dtype under pandas>=2.3
# (`pd.arrays.StringArray` / `ArrowStringArray`), and anndata refuses to serialize that dtype unless
# told which format to use:
#
#   RuntimeError: `anndata.settings.allow_write_nullable_strings` is None and
#   `pd.options.future.infer_string` is False. Opt-in to writing these arrays by toggling either
#   setting to True, or make anndata attempt to write the non-nullable format supported by
#   anndata < 0.11 by setting `allow_write_nullable_strings` to False.
#
# `False` -- the legacy, plain-string format, readable by any anndata version -- is used here on the
# assumption that these string columns are always fully populated (true of every real Xenium table
# inspected so far), rather than the newer format that needs anndata>=0.11 to read back: spatialdata
# itself only requires anndata>=0.9.1, so a store written in the newer format may not be readable by
# whoever opens it outside this repo's own pinned environment. `False` raises if a string column ever
# does have a missing value -- a real failure to fix then, not one to route around here.
ad.settings.allow_write_nullable_strings = False

# -------------------------------------------------------------------------------

# Copied verbatim from `phoenix_demo.ipynb`. Reproducibility-critical: the shipped weights were
# trained against exactly this preprocessing, so these are module constants and not arguments.
IMAGE_TRANSFORM = v2.Compose(
    [
        v2.Resize((224, 224), InterpolationMode.BICUBIC),
        v2.CenterCrop((224, 224)),
        v2.ToTensor(),
        v2.Normalize(
            (0.707223, 0.578729, 0.703617),
            (0.211883, 0.230117, 0.177517),
        ),
    ]
)

# The *published* inference configuration, as used by the notebook and the README. It deliberately
# differs from `FlowTransformerConfig`'s dataclass defaults, which describe a larger model that the
# shipped checkpoint would not load into. `FlowTransformerModel` stores this without mutating it, so
# sharing one instance across calls is safe.
MODEL_CONFIG = FlowTransformerConfig(
    d_genes=1,
    d_image=1536,
    d_model=512,
    d_cross=512,
    n_heads=8,
    n_layers=8,
    qkv_bias=False,
    ffn_bias=False,
    ffn_mult=4,
    attn_drop=0.0,
    proj_drop=0.0,
    n_classes=0,
    cls_drop=0.1,
    checkpoint=False,
)

# The frozen vision encoder. Built with `pretrained=False` because its weights ship inside the single
# `flow_model.pth` checkpoint rather than being downloaded separately.
VISION_MODEL = "vit_giant_patch14_reg4_dinov2"

# Element *types* read from each store. Patch extraction needs `images` and `shapes`, the prediction
# needs `tables`; `points` (the transcripts frame) and `labels` (the segmentation pyramids) are never
# touched, and skipping them cuts the per-store read substantially on large slides.
# NOTE: the spelling is plural. `sd.read_zarr`'s docstring says "table", and that spelling silently
# returns a store with *no* tables rather than raising.
READ_SELECTION = ("images", "shapes", "tables")


def load_model(weights: str | Path, device: str | torch.device) -> FlowTransformerModel:
    """
    Build the flow transformer with its frozen vision encoder and load the published weights.

    Uses the optimized `phoenix.models.flow_llama3` implementation (apex, flash-attn, xformers). The
    pure-torch `phoenix.models.flow_simple` variant is mathematically equivalent and runs anywhere,
    including CPU; this one requires an sm_80+ CUDA device.

    Parameters
    ----------
    weights
        Path to a `flow_model.pth` checkpoint. It carries the DINOv2 encoder weights as well as the
        flow transformer's, which is why `strict=True` succeeds against a `pretrained=False` encoder.
    device
        CUDA device the model is moved to, e.g. ``"cuda:0"``. `FlashAttention.forward` hardcodes
        `device_type="cuda"` autocast and calls `flash_attn_func`, so a CPU device fails here.

    Returns
    -------
    The model in eval mode on `device`.
    """
    vision_model = timm.create_model(
        VISION_MODEL,
        pretrained=False,
        img_size=224,
        num_classes=0,
        global_pool="token",
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model = FlowTransformerModel(MODEL_CONFIG, vision_model=vision_model)

    # `map_location="cpu"` then `.to(device)`, as in the notebook: the checkpoint is ~4.5 GB and
    # loading it straight onto a GPU would need that much free device memory in one allocation.
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model.eval().to(device)


def predict_slide(
    zarr_path: str | Path,
    model: FlowTransformerModel,
    gene_list: Sequence[str],
    stats: Mapping[str, np.ndarray],
    *,
    table_key: str = "table",
    pred_key: str = "pred_table",
    batch_size: int = 128,
    num_workers: int = 0,
    patch_size: int = 224,
    target_mpp: float = 0.5,
    atol: float = 1e-1,
    rtol: float = 1e-1,
    fast: bool = False,
    seed: int = 0,
    overwrite: bool = False,
) -> int | None:
    """
    Predict gene expression for one store and write it back as a second table.

    The prediction is written into the *same* zarr store as `pred_key`, carrying `table_key`'s
    `obs`, `var`, `obsm["spatial"]` and `uns["spatialdata_attrs"]`. Copying the spatialdata attrs
    makes the new table annotate the same region element as the ground-truth table, which is what
    lets `spatialdata_plot` render either of them by gene name afterwards (pass `table_name=` to
    disambiguate, since two tables then annotate the same region).

    Parameters
    ----------
    zarr_path
        Path to a SpatialData `.zarr` store holding `he_image`, `nucleus_boundaries` and `table_key`.
    model
        Model from `load_model`. Its device determines where sampling runs.
    gene_list
        The gene panel, in panel order. Also fixes the column order of the written table.
    stats
        Mapping with ``"mean"`` and ``"std"`` entries, ``(n_genes,)`` each, in panel order; used by
        `FlowPipeline` to de-normalise predictions.
    table_key
        Key of the ground-truth table to take cells and metadata from.
    pred_key
        Key the predicted table is written under.
    batch_size
        Cells per forward pass.
    num_workers
        `DataLoader` worker processes. Zero (the default) reads patches in the main process.
    patch_size
        Side length, in pixels at `target_mpp`, of the patch extracted per cell.
    target_mpp
        Target resolution in microns per pixel.
    atol
        Absolute tolerance of the adaptive ODE solver.
    rtol
        Relative tolerance of the adaptive ODE solver.
    fast
        Whether to use the K/V-caching sampler. Slightly different numerically.
    seed
        Seed applied *before* this slide's sampling, so its output does not depend on how many other
        slides the calling process handled first.
    overwrite
        Whether to replace an existing `pred_key`. When false, a store that already has one is
        skipped.

    Returns
    -------
    Number of cells written, or `None` if the store already had `pred_key` and was skipped.
    """
    name = Path(zarr_path).name

    # Seed per slide, not per process: the sampler draws its initial noise from the global RNG, so
    # without this a slide's prediction would depend on its position in the work queue.
    torch.manual_seed(seed)

    sdata = sd.read_zarr(zarr_path, selection=READ_SELECTION)
    replacing = pred_key in sdata.tables
    if replacing and not overwrite:
        logger.info("%s: '%s' already present, skipping", name, pred_key)
        return None

    dataset = SpatialDataset(
        zarr_path=sdata,
        table_type=table_key,
        gene_list=list(gene_list),
        patch_size=patch_size,
        target_mpp=target_mpp,
        image_transform=IMAGE_TRANSFORM,
        adata_transform=None,
    )
    # `shuffle=False` is load-bearing, not a default: cells are matched to predictions by row
    # position alone -- there is no cell-id join anywhere downstream.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info("%s: sampling %d cells over %d genes", name, len(dataset), len(gene_list))
    pipeline = FlowPipeline(
        model=model,
        stats=stats,
        t_0=0.0,
        t_1=1.0,
        atol=atol,
        rtol=rtol,
        fast=fast,
    )
    # `coords_list` is discarded: it is the shape-space centroids, already in `obsm["spatial"]`.
    gex_pred, _ = pipeline(list(gene_list), dataloader)

    adata = dataset.adata
    expected = (adata.n_obs, len(gene_list))
    if gex_pred.shape != expected:
        # Fail loudly: a wrong shape here would silently mis-assign every gene in the written table.
        raise ValueError(f"{name}: predicted {gex_pred.shape}, expected {expected}")

    if replacing:
        # Delete first rather than relying on `overwrite=True` alone: `write_element` writes into the
        # existing zarr group in place, so a previous table with different columns could leave stale
        # arrays behind. Deleting here -- after sampling succeeded -- means a crash mid-inference
        # leaves the old prediction intact.
        sdata.delete_element_from_disk(pred_key)

    # Wrap the prediction in an AnnData carrying table's metadata: `obs`/`var` verbatim (panel-ordered
    # columns, original row order), plus `obsm["spatial"]` and `uns["spatialdata_attrs"]` so the new
    # table annotates the same region as `table` and `spatialdata_plot` can render either by name.
    pred = ad.AnnData(X=gex_pred.astype(np.float32), obs=adata.obs.copy(), var=adata.var.copy())
    pred.obsm["spatial"] = adata.obsm["spatial"].copy()
    pred.uns["spatialdata_attrs"] = dict(adata.uns["spatialdata_attrs"])
    sdata[pred_key] = TableModel.parse(pred)
    # `write_element` re-consolidates the store's metadata itself, walking what is on disk rather
    # than what was read, so the unread `points`/`labels` elements survive intact.
    sdata.write_element(pred_key, overwrite=overwrite)
    logger.info("%s: wrote '%s' with %d cells", name, pred_key, adata.n_obs)

    return adata.n_obs


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
        description="Predict spatial gene expression for a list of SpatialData zarr stores, using "
        "the optimized apex/flash-attn/xformers flow transformer (CUDA sm_80+ only).",
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
        help="CUDA device, e.g. 'cuda:0'; default is 'cuda' if visible. There is no CPU path.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the K/V-caching sampler (slightly different numerics)",
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
    Work out which CUDA device to run on.

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
    Process exit status: non-zero if any slide failed, or 2 if no CUDA device is available.
    """
    args = parse_args(argv)
    device = resolve_device(args.device)
    configure_logging(device)

    if not device.startswith("cuda"):
        # Fail before touching the model: `FlashAttention.forward` hardcodes a CUDA autocast and
        # calls `flash_attn_func`, so running this script on CPU fails deep inside the first forward
        # pass with an opaque CUDA error instead of a clear one.
        logger.error("device '%s' is not CUDA; flow_llama3 has no CPU path (use run_inference.py)", device)
        return 2

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
