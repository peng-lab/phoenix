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

Each store can be split across several GPUs with Lightning DDP: launch one process per GPU (under
`srun --ntasks-per-node=N`, which Lightning reads from the SLURM environment) and pass `--devices N`
(and `--num-nodes`). Every rank reads the store and predicts every N-th cell; rank 0 puts the rows
back in table order and writes the same `pred_table` a single-GPU run writes. Each rank seeds its
sampler with `seed + rank`, so with one device the output is bit-identical to the unsharded script,
and with several it differs from it only as much as a different seed would. With several devices a
failed store aborts the job instead of being skipped (the other ranks would hang at the next
collective); stores that already hold a `pred_table` are skipped, so resubmitting resumes.

Usage
-----
    python run_inference_flash.py \
        --weights /path/to/flow_model.pth \
        --panel /path/to/xenium_human_multi.npy \
        --stats /path/to/stats_table.npz \
        store_a.zarr store_b.zarr store_c.zarr

    srun --ntasks-per-node=4 --gres=gpu:4 python run_inference_flash.py --devices 4 \
        --weights ... --panel ... --stats ... store_a.zarr

© Peng Lab / Helmholtz Munich
"""

import argparse
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import spatialdata as sd
import timm
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import BasePredictionWriter, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
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

# Timeout of the gloo group that gathers each rank's predictions on rank 0. Generous because ranks
# wait there for the slowest one (adaptive step counts vary per batch) and for rank 0's zarr write.
GATHER_TIMEOUT = timedelta(hours=2)

# Batches between progress-bar updates. Lightning's default Rich bar redraws in place and prints
# only its final state when stdout is a file, so a SLURM log shows nothing until the store is done;
# the tqdm bar appends a line per update instead.
PROGRESS_REFRESH_BATCHES = 10


def load_model(weights: str | Path) -> FlowTransformerModel:
    """
    Build the flow transformer with its frozen vision encoder and load the published weights.

    Uses the optimized `phoenix.models.flow_llama3` implementation (apex, flash-attn, xformers). The
    pure-torch `phoenix.models.flow_simple` variant is mathematically equivalent and runs anywhere,
    including CPU; this one requires an sm_80+ CUDA device (`FlashAttention.forward` hardcodes
    `device_type="cuda"` autocast and calls `flash_attn_func`).

    Parameters
    ----------
    weights
        Path to a `flow_model.pth` checkpoint. It carries the DINOv2 encoder weights as well as the
        flow transformer's, which is why `strict=True` succeeds against a `pretrained=False` encoder.

    Returns
    -------
    The model in eval mode on CPU; `pl.Trainer` moves it to each rank's GPU.
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

    # `map_location="cpu"`, as in the notebook: the checkpoint is ~4.5 GB and loading it straight
    # onto a GPU would need that much free device memory in one allocation.
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model.eval()


class PhoenixPredictor(pl.LightningModule):
    """
    Lightning wrapper that lets `pl.Trainer.predict` drive `FlowPipeline.sample_batch`.

    Parameters
    ----------
    model
        Model from `load_model`. `pl.Trainer` moves it to the rank's device.
    stats
        Mapping with ``"mean"`` and ``"std"`` entries, ``(n_genes,)`` each, in panel order.
    n_genes
        Panel size; sizes the initial noise.
    atol
        Absolute tolerance of the adaptive ODE solver.
    rtol
        Relative tolerance of the adaptive ODE solver.
    fast
        Whether to use the K/V-caching sampler. Slightly different numerically.
    seed
        Base seed. Each store's sampling is seeded with ``seed + rank`` at the start of `trainer.predict`,
        so a store's output does not depend on how many stores came before it, and ranks do not share noise.
    """

    def __init__(
        self,
        model: FlowTransformerModel,
        stats: Mapping[str, np.ndarray],
        n_genes: int,
        atol: float,
        rtol: float,
        fast: bool,
        seed: int,
    ):
        super().__init__()
        self.model = model
        self.stats = stats
        self.n_genes = n_genes
        self.atol = atol
        self.rtol = rtol
        self.fast = fast
        self.seed = seed

    def on_predict_start(self) -> None:
        """Seed this rank and build the pipeline; it captures the model's device, so it must follow the move to the GPU."""
        # Seed per store, not per process, and per rank: ranks with the same seed would draw the same
        # noise for neighbouring cells. With one rank this is plain `seed`.
        torch.manual_seed(self.seed + self.global_rank)
        self.pipeline = FlowPipeline(
            model=self.model,
            stats=self.stats,
            t_0=0.0,
            t_1=1.0,
            atol=self.atol,
            rtol=self.rtol,
            fast=self.fast,
        )

    def predict_step(self, batch: tuple, batch_idx: int) -> np.ndarray:
        """Sample one batch of ``(image, coords)``; returns ``(batch, n_genes)`` in normalized space."""
        return self.pipeline.sample_batch(batch[0], self.n_genes)


class OrderedPredictionGatherer(BasePredictionWriter):
    """
    Collect every rank's predictions on rank 0, in table order.

    Lightning's distributed sampler gives each rank an interleaved subset of the cells, so the rows
    have to be put back where they came from (the `batch_indices` of each batch) before the result
    matches the single-GPU one. This callback only assembles the array; `predict_slide` writes the table.

    Attributes
    ----------
    prediction
        De-normalized predictions, ``(n_cells, n_genes)`` in table order. Set on rank 0 only.
    """

    def __init__(self):
        super().__init__(write_interval="epoch")
        self.prediction: np.ndarray | None = None
        self._group = None

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices) -> None:
        """
        Gather the ranks' rows on rank 0 and restore table order.

        Parameters
        ----------
        trainer
            The running trainer.
        pl_module
            The `PhoenixPredictor`, whose pipeline de-normalizes the result.
        predictions
            This rank's per-batch outputs of `predict_step`.
        batch_indices
            This rank's per-batch table row indices, one list per dataloader (there is one).
        """
        local = (np.concatenate(batch_indices[0]), np.concatenate(predictions))

        if trainer.world_size == 1:
            parts = [local]
        else:
            # gloo, so the (up to ~1 GB) payload moves over host memory and not through the GPUs
            self._group = self._group or dist.new_group(backend="gloo", timeout=GATHER_TIMEOUT)
            parts = [None] * trainer.world_size if trainer.is_global_zero else None
            dist.gather_object(local, parts, dst=0, group=self._group)
        if not trainer.is_global_zero:
            return

        indices = np.concatenate([part[0] for part in parts])
        gathered = np.concatenate([part[1] for part in parts])
        # a row filled twice or never would silently mis-assign cells in the written table
        if not np.array_equal(np.sort(indices), np.arange(len(indices))):
            raise ValueError("gathered row indices are not a permutation of the table rows")

        ordered = np.empty_like(gathered)
        ordered[indices] = gathered
        self.prediction = pl_module.pipeline.denormalize(ordered)


def predict_slide(
    zarr_path: str | Path,
    trainer: pl.Trainer,
    predictor: PhoenixPredictor,
    gatherer: OrderedPredictionGatherer,
    gene_list: Sequence[str],
    *,
    table_key: str = "table",
    pred_key: str = "pred_table",
    batch_size: int = 128,
    num_workers: int = 0,
    patch_size: int = 224,
    target_mpp: float = 0.5,
    overwrite: bool = False,
) -> int | None:
    """
    Predict gene expression for one store, across all of the trainer's ranks, and write it back as a second table.

    Every rank must call this with the same store. The prediction is assembled and written by rank 0 only,
    into the *same* zarr store as `pred_key`, carrying `table_key`'s
    `obs`, `var`, `obsm["spatial"]` and `uns["spatialdata_attrs"]`. Copying the spatialdata attrs
    makes the new table annotate the same region element as the ground-truth table, which is what
    lets `spatialdata_plot` render either of them by gene name afterwards (pass `table_name=` to
    disambiguate, since two tables then annotate the same region).

    Parameters
    ----------
    zarr_path
        Path to a SpatialData `.zarr` store holding `he_image`, `nucleus_boundaries` and `table_key`.
    trainer
        Trainer that runs the prediction; its world size sets how many ranks share the store.
    predictor
        Wrapper around the model and the sampler settings.
    gatherer
        Callback registered on `trainer` that assembles the ranks' predictions on rank 0.
    gene_list
        The gene panel, in panel order. Also fixes the column order of the written table.
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
    overwrite
        Whether to replace an existing `pred_key`. When false, a store that already has one is
        skipped.

    Returns
    -------
    Number of cells predicted, or `None` if the store already had `pred_key` and was skipped.
    """
    name = Path(zarr_path).name

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
    # a wrong native pixel size silently changes the patch's field of view; make it visible in the logs
    crop_px = int(patch_size * target_mpp / dataset.native_mpp)
    logger.info(
        "%s: native pixel size %.4f um/px, %d px crop covers %.1f um",
        name,
        dataset.native_mpp,
        crop_px,
        crop_px * dataset.native_mpp,
    )
    # Lightning swaps in its distributed sampler under DDP, which hands each rank an interleaved
    # subset of the cells; `shuffle=False` keeps that assignment (and the single-rank order) fixed.
    # Cells are matched to predictions by table row alone -- `gatherer` restores the order from the
    # batch indices, and there is no cell-id join anywhere downstream.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info("%s: sampling %d cells over %d genes", name, len(dataset), len(gene_list))
    trainer.predict(predictor, dataloader, return_predictions=False)

    adata = dataset.adata
    if trainer.is_global_zero:
        gex_pred = gatherer.prediction
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
    parser.add_argument("--seed", type=int, default=0, help="seed, applied per slide (plus the rank)")

    parser.add_argument("--devices", type=int, default=1, help="GPUs (= SLURM tasks) per node sharing each store")
    parser.add_argument("--num-nodes", type=int, default=1, help="nodes sharing each store")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the K/V-caching sampler (slightly different numerics)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --pred-key")

    return parser.parse_args(argv)


def configure_logging() -> None:
    """
    Send this process's log to stdout, tagged with its rank.

    Only rank 0 logs progress; the other ranks log warnings and errors, so a store is not reported
    once per GPU.
    """
    rank = rank_zero_only.rank  # read from the SLURM/torchrun environment, so it is known before the Trainer exists
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    """
    Predict every given store, sharing each across the requested GPUs, and report what happened.

    Parameters
    ----------
    argv
        Argument list; defaults to `sys.argv[1:]`.

    Returns
    -------
    Process exit status: non-zero if any slide failed, or 2 if no CUDA device is available.
    """
    args = parse_args(argv)
    configure_logging()

    if not torch.cuda.is_available():
        # Fail before touching the model: `FlashAttention.forward` hardcodes a CUDA autocast and
        # calls `flash_attn_func`, so running this script on CPU fails deep inside the first forward
        # pass with an opaque CUDA error instead of a clear one.
        logger.error("no CUDA device is available; flow_llama3 has no CPU path (use run_inference.py)")
        return 2

    for label, path in (("weights", args.weights), ("panel", args.panel), ("stats", args.stats)):
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            return 2

    logger.info(
        "%d store(s) on %d device(s) x %d node(s); seed=%d atol=%g rtol=%g batch_size=%d num_workers=%d fast=%s",
        len(args.stores),
        args.devices,
        args.num_nodes,
        args.seed,
        args.atol,
        args.rtol,
        args.batch_size,
        args.num_workers,
        args.fast,
    )

    logger.info("loading %s", args.weights)
    gene_list = list(np.load(args.panel))
    predictor = PhoenixPredictor(
        load_model(args.weights),
        np.load(args.stats),
        n_genes=len(gene_list),
        atol=args.atol,
        rtol=args.rtol,
        fast=args.fast,
        seed=args.seed,
    )
    gatherer = OrderedPredictionGatherer()
    # `inference_mode=False`: `run_flow` already runs under `torch.no_grad`, as before; "32-true" because
    # flow_llama3 applies its own bf16 autocast and a Lightning precision plugin would stack on top of it.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices * args.num_nodes > 1 else "auto",
        precision="32-true",
        inference_mode=False,
        callbacks=[gatherer, TQDMProgressBar(refresh_rate=PROGRESS_REFRESH_BATCHES)],
        logger=False,
        enable_checkpointing=False,
    )

    written, skipped, failed = [], [], []
    for path in args.stores:
        name = path.name
        start = time.perf_counter()
        try:
            n_cells = predict_slide(
                path,
                trainer,
                predictor,
                gatherer,
                gene_list,
                table_key=args.table_key,
                pred_key=args.pred_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                patch_size=args.patch_size,
                target_mpp=args.target_mpp,
                overwrite=args.overwrite,
            )
        except Exception:
            logger.exception("%s: failed", name)
            if trainer.world_size > 1:
                # the other ranks would hang at the next collective; exiting non-zero lets
                # `srun --kill-on-bad-exit=1` tear the step down, and a rerun resumes after the finished stores
                raise
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
