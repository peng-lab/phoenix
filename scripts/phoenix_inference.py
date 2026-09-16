"""
Core logic for batch Phoenix inference over SpatialData `.zarr` stores.

Everything numeric here is the `phoenix_demo.ipynb` code path unchanged: the image normalisation
constants, the published model configuration, the panel-ordered gene axis, the ODE sampler and the
de-normalisation all come from `phoenix` itself. Given the same store, weights, statistics, seed and
solver tolerance, this module reproduces what the notebook produces -- it is a driver, not a new
model.

The one piece of state this module owns is the gene panel's *order*, which is load-bearing:
`FlowTransformerModel` encodes gene identity positionally (`px_embedding` indexes gene slots), so the
panel order must be identical across the dataset's column subset, the sampler's noise tensor, the
columns of `gex_pred` and the `mean`/`std` vectors in the statistics file.

Driven by `run_inference.py`.

© Peng Lab / Helmholtz Munich
"""

import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import spatialdata as sd
import timm
import torch
import zarr
from spatialdata.models import TableModel
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2

from phoenix.datasets.zarr_dataset import SpatialDataset
from phoenix.helpers.inference import FlowPipeline
from phoenix.models.flow_simple import FlowTransformerConfig, FlowTransformerModel

logger = logging.getLogger(__name__)

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

    Uses the pure-torch `phoenix.models.flow_simple` implementation. The optimized
    `phoenix.models.flow_llama3` variant is mathematically equivalent but needs apex, flash-attn and
    xformers, which are deliberately absent from this environment.

    Parameters
    ----------
    weights
        Path to a `flow_model.pth` checkpoint. It carries the DINOv2 encoder weights as well as the
        flow transformer's, which is why `strict=True` succeeds against a `pretrained=False` encoder.
    device
        Device the model is moved to, e.g. ``"cuda:0"`` or ``"cpu"``.

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
        Whether to use the K/V-caching sampler. CUDA-only and slightly different numerically.
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
