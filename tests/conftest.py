from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def adata():
    adata = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    adata.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)
    adata.var_names = ["PECAM1", "MMRN2"]
    adata.obsm["spatial"] = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
    return adata


def make_synthetic_store(
    path: Path,
    n_cells: int = 32,
    genes: Sequence[str] = ("PECAM1", "MMRN2", "MYH11", "SFRP2"),
    image_size: int = 512,
    seed: int = 0,
) -> Path:
    """
    Write a minimal SpatialData ``.zarr`` store that `SpatialDataset` can consume.

    The store holds a multiscale ``he_image``, circular ``nucleus_boundaries`` and a
    ``table`` of Poisson counts annotating them, all under identity transforms, so the
    native resolution resolves to 1.0 micron per pixel. Cells are placed away from the
    image border except the last one, which sits on the left edge so its patch is clipped
    along x only and comes out non-square (a cell in the corner would clip both axes to an
    empty, square patch that escapes the dataset's blank-patch fallback).

    Parameters
    ----------
    path
        Where to write the store.
    n_cells
        Number of cells (rows of the table, nucleus shapes).
    genes
        Gene panel of the table, in this order.
    image_size
        Side length, in pixels, of the square ``he_image`` at scale 0.
    seed
        Seed of the ``numpy`` generator drawing coordinates, pixels and counts.

    Returns
    -------
    The path the store was written to.
    """
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import Image2DModel, ShapesModel, TableModel

    rng = np.random.default_rng(seed)
    genes = list(genes)

    xy = rng.uniform(64, image_size - 64, size=(n_cells, 2))
    xy[-1] = (2.0, image_size / 2)

    image = rng.integers(0, 256, size=(3, image_size, image_size), dtype=np.uint8)
    he_image = Image2DModel.parse(image, dims=("c", "y", "x"), scale_factors=[2])
    nuclei = ShapesModel.parse(xy, geometry=0, radius=5.0)

    counts = csr_matrix(rng.poisson(2.0, size=(n_cells, len(genes))).astype(np.float32))
    obs = pd.DataFrame(
        {
            "instance_id": np.arange(n_cells),
            "region": pd.Categorical(["nucleus_boundaries"] * n_cells),
        },
        index=np.arange(n_cells).astype(str),
    )
    adata = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = xy
    table = TableModel.parse(adata, region="nucleus_boundaries", region_key="region", instance_key="instance_id")

    sdata = sd.SpatialData(
        images={"he_image": he_image}, shapes={"nucleus_boundaries": nuclei}, tables={"table": table}
    )
    sdata.write(path)
    return path


@pytest.fixture
def synthetic_store(tmp_path):
    return make_synthetic_store(tmp_path / "store.zarr")
