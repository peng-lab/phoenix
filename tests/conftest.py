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
    attrs: dict | None = None,
    he_to_global=None,
    shape_to_global=None,
    margin: int = 64,
    blob_cell: int | None = None,
) -> Path:
    """
    Write a minimal SpatialData ``.zarr`` store that `SpatialDataset` can consume.

    The store holds a multiscale ``he_image``, circular ``nucleus_boundaries`` and a
    ``table`` of Poisson counts annotating them. By default both elements sit under
    identity transforms. Cells are drawn in he-pixel space and mapped into shape space
    through the two transforms (rounded to integers, as `SpatialDataset` truncates them),
    so a layout with non-trivial ``he_to_global`` / ``shape_to_global`` stays
    self-consistent. Cells are placed away from the image border except the last one,
    which sits on the left edge so its patch is clipped along x only and comes out
    non-square (a cell in the corner would clip both axes to an empty, square patch that
    escapes the dataset's blank-patch fallback).

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
    attrs
        Entries written to ``sdata.attrs`` (``spatialdata_io_reader``, ``source_mpp``,
        ``source_he_mpp``).
    he_to_global
        Transformation of ``he_image`` to ``"global"``; identity when `None`.
    shape_to_global
        Transformation of ``nucleus_boundaries`` to ``"global"``; identity when `None`.
    margin
        Minimum distance, in he pixels, of a cell (the border cell excepted) from the image edge.
    blob_cell
        If given, the image is black except for a white 7x7 px blob on this cell's he pixel,
        which lets a test locate that cell inside the extracted patch.

    Returns
    -------
    The path the store was written to.
    """
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import Image2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity

    rng = np.random.default_rng(seed)
    genes = list(genes)
    he_to_global = Identity() if he_to_global is None else he_to_global
    shape_to_global = Identity() if shape_to_global is None else shape_to_global

    xy_pixel = rng.uniform(margin, image_size - margin, size=(n_cells, 2))
    xy_pixel[-1] = (2.0, image_size / 2)

    # he pixel -> global -> shape, in the (y, x) axis order `SpatialDataset` uses
    pixel_global = he_to_global.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
    shape_global = shape_to_global.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
    yx1 = np.c_[xy_pixel[:, 1], xy_pixel[:, 0], np.ones(n_cells)]
    xy = np.round((yx1 @ (np.linalg.inv(shape_global) @ pixel_global).T)[:, [1, 0]])

    if blob_cell is None:
        image = rng.integers(0, 256, size=(3, image_size, image_size), dtype=np.uint8)
    else:
        image = np.zeros((3, image_size, image_size), dtype=np.uint8)
        blob = np.linalg.inv(pixel_global) @ shape_global @ np.array([xy[blob_cell, 1], xy[blob_cell, 0], 1.0])
        y_blob, x_blob = int(round(blob[0])), int(round(blob[1]))
        image[:, y_blob - 3 : y_blob + 4, x_blob - 3 : x_blob + 4] = 255
    he_image = Image2DModel.parse(
        image, dims=("c", "y", "x"), scale_factors=[2], transformations={"global": he_to_global}
    )
    nuclei = ShapesModel.parse(xy, geometry=0, radius=5.0, transformations={"global": shape_to_global})

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
    sdata.attrs.update(attrs or {})
    sdata.write(path)
    return path


# an h&e-only store at 1 micron per pixel, the pixel size `SpatialDataset` used to assume
@pytest.fixture
def synthetic_store(tmp_path):
    return make_synthetic_store(tmp_path / "store.zarr", attrs={"spatialdata_io_reader": "he", "source_he_mpp": 1.0})
