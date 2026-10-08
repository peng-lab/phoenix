"""
Spatial transcriptomics dataset based on SpatialData.

© Peng Lab / Helmholtz Munich
"""

import warnings
from pathlib import Path

import numpy as np
import spatialdata as sd
import torch
from PIL import Image
from scipy.sparse import csr_matrix, issparse
from spatialdata.transformations import get_transformation
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# ------------------------------------------------------------------------------------------


class SpatialDataset(Dataset):
    """
    Spatial transcriptomics dataset backed by a SpatialData ``.zarr`` store.

    Extracts an anndata table and the paired H&E image (``he_image``) and nucleus
    segmentation (``nucleus_boundaries``) elements from the store, and generates
    image/expression pairs on access by re-projecting each cell's shape-space
    coordinates into pixel space.

    Parameters
    ----------
    zarr_path
        Path to a SpatialData ``.zarr`` store, or an already-read ``SpatialData``
        object (e.g. from ``sd.read_zarr(path, selection=...)``) used as is.
    table_type
        Key of the anndata table to read from the store (used when `adata_transform`
        is not given).
    gene_list
        Genes to subset the anndata table to.
    patch_size
        Side length, in pixels at `target_mpp`, of the image patch extracted per cell.
    target_mpp
        Target resolution in microns per pixel; defaults to the store's native
        resolution when `None`.
    adata_transform
        Optional callable applied to the raw anndata table (`self.sdata[table_type]`)
        in place of the default `get_adata` extraction.
    image_transform
        Optional torchvision-style transform applied to each extracted image patch.
    native_mpp
        Pixel size of the H&E image, in microns per pixel. Determined from the store's
        attributes and transformations when `None` (see `get_native`); a patch covers
        ``patch_size * target_mpp`` microns only if this is the true pixel size.
    """

    def __init__(
        self,
        zarr_path: str | Path | sd.SpatialData,
        table_type: str,
        gene_list: list,
        patch_size: int = 224,
        target_mpp: float = 0.5,
        adata_transform: Compose | None = None,
        image_transform: Compose | None = None,
        native_mpp: float | None = None,
    ):
        # read zarr file with spatialdata, unless an already-read store is given
        self.sdata = zarr_path if isinstance(zarr_path, sd.SpatialData) else sd.read_zarr(zarr_path)

        if adata_transform:
            adata = self.sdata[table_type]
            self.adata = adata_transform(adata)
        else:
            self.adata = self.get_adata(table_type, gene_list)

        # get the gene expression matrix
        # self.gene_matrix = self.adata.X.tocsr()
        self.gene_matrix = csr_matrix(self.adata.X)

        # setup all data transformations
        self.adata_transform = adata_transform
        self.image_transform = image_transform

        # data -> global
        self.pixel_to_global = get_transformation(
            self.sdata["he_image"],
            to_coordinate_system="global",
        ).to_affine_matrix(
            input_axes=("y", "x"),
            output_axes=("y", "x"),
        )
        self.shape_to_global = get_transformation(
            self.sdata["nucleus_boundaries"],
            to_coordinate_system="global",
        ).to_affine_matrix(
            input_axes=("y", "x"),
            output_axes=("y", "x"),
        )

        # global -> data
        self.global_to_pixel = np.linalg.inv(self.pixel_to_global)
        self.global_to_shape = np.linalg.inv(self.shape_to_global)

        # store all image hyperparameters
        self.patch_size = patch_size
        self.native_mpp = self.get_native() if native_mpp is None else native_mpp
        self.target_mpp = self.native_mpp if target_mpp is None else target_mpp

    def get_native(self):
        """
        Determine the pixel size of the H&E image, in microns per pixel.

        What a "global" unit is in microns depends on how the store was built, which is
        recorded in ``sdata.attrs["spatialdata_io_reader"]``:

        - ``"xenium"``: global is the morphology pixel grid, so one global unit is
          ``attrs["source_mpp"]`` microns.
        - ``"he"``: global is the H&E pixel grid, so the pixel size is
          ``attrs["source_he_mpp"]`` directly, whatever the transformations are.
        - no reader attribute: one global unit is read off the ``nucleus_boundaries``
          transformation, which assumes the shapes are stored in microns. An identity
          transformation cannot tell microns from pixels, so it is rejected.

        Returns
        -------
        Native resolution in microns per pixel.

        Raises
        ------
        ValueError
            If the attribute the store's reader requires is missing, or the store has
            neither a reader attribute nor a micron-scaled ``nucleus_boundaries``.
        """
        attrs, scale, affine = self.sdata.attrs, self.shape_to_global, self.pixel_to_global
        reader = attrs.get("spatialdata_io_reader")

        required = {"xenium": "source_mpp", "he": "source_he_mpp"}.get(reader)
        if required is not None and required not in attrs:
            raise ValueError(f"store read by '{reader}' has no '{required}' attribute; pass native_mpp explicitly")

        if reader == "he":
            return float(attrs["source_he_mpp"])

        if reader == "xenium":
            micron_per_global = attrs["source_mpp"]
        elif np.allclose(scale, np.eye(3)):
            raise ValueError(
                "cannot determine the pixel size: the store has no 'spatialdata_io_reader' attribute and an "
                "identity 'nucleus_boundaries' transformation; set 'source_he_mpp' or pass native_mpp explicitly"
            )
        else:
            micron_per_global = 2 / (scale[0][0] + scale[1][1])

        scale_x = np.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2)
        scale_y = np.sqrt(affine[0, 1] ** 2 + affine[1, 1] ** 2)

        global_per_pixel = (scale_x + scale_y) / 2.0
        micron_per_pixel = global_per_pixel * micron_per_global

        # the two sources of a xenium store's pixel size should agree; a large gap points to a mis-set attribute
        he_mpp = attrs.get("source_he_mpp")
        if reader == "xenium" and he_mpp is not None and abs(micron_per_pixel / he_mpp - 1) > 0.05:
            warnings.warn(
                f"pixel size {micron_per_pixel:.4f} um/px derived from 'source_mpp' differs from "
                f"'source_he_mpp' = {he_mpp:.4f} um/px by more than 5%",
                stacklevel=2,
            )

        return micron_per_pixel

    def get_adata(self, table_type: str, gene_list: list):
        """
        Read an anndata table from the store and subset it to `gene_list`.

        Parameters
        ----------
        table_type
            Key of the anndata table to read from the store.
        gene_list
            Genes to subset the table to.

        Returns
        -------
        The subsetted anndata table.
        """
        # extract table as anndata object
        adata = self.sdata[table_type]
        # subset adata to genes in panel
        adata = adata[:, gene_list].copy()
        return adata

    def get_patch(self, x_center_shape: int, y_center_shape: int):
        """
        Extract an image patch centered on a cell's shape-space coordinates.

        Parameters
        ----------
        x_center_shape
            Cell center, x coordinate, in shape space.
        y_center_shape
            Cell center, y coordinate, in shape space.

        Returns
        -------
        The extracted patch as a ``(patch_size, patch_size, 3)`` uint8 array.
        """
        # extract image as xarray object
        he_image = self.sdata["he_image"]

        # scale image to correct size
        scaling_factor = self.target_mpp / self.native_mpp
        patch_size = int(self.patch_size * scaling_factor)

        # used for corner coordinates
        half_size = patch_size // 2

        # center coordinates (pixels)
        shape_coords = np.array([y_center_shape, x_center_shape, 1.0])
        pixel_coords = self.global_to_pixel @ (self.shape_to_global @ shape_coords)

        # round coordinates to integer
        y_center_pixel, x_center_pixel, _ = pixel_coords
        x_center, y_center = int(round(x_center_pixel)), int(round(y_center_pixel))

        # corner coordinates (pixels)
        x_start, y_start = x_center - half_size, y_center - half_size
        x_end, y_end = x_center + half_size, y_center + half_size

        # patch image at scale 0
        patch = he_image.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
        patch = patch["/scale0"].ds["image"].values

        # transform image patch
        patch = patch.transpose(1, 2, 0)
        patch = patch.astype(np.uint8)

        return patch

    def __getitem__(self, idx: int):
        """
        Return the image patch and gene expression vector for a single cell.

        Falls back to a blank (all-zero) 224x224x3 patch when the extracted patch
        isn't square (e.g. a cell near the image border).

        Parameters
        ----------
        idx
            Index into the anndata table.

        Returns
        -------
        Tuple of the (optionally transformed) image patch and the cell's
        ``[x_center_shape, y_center_shape]`` coordinates.
        """
        # gene expression
        values = self.gene_matrix[idx]
        values = values.toarray() if issparse(values) else values
        values = torch.tensor(values, dtype=torch.float32)

        # patch coordinates
        x_center_shape = int(self.adata.obsm["spatial"][idx, 0])
        y_center_shape = int(self.adata.obsm["spatial"][idx, 1])

        # image patch / tile
        image = self.get_patch(x_center_shape, y_center_shape)
        if image.shape[0] != image.shape[1]:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # image transforms
        image = Image.fromarray(image)
        if self.image_transform:
            image = self.image_transform(image)

        return image, np.array([x_center_shape, y_center_shape])

    def __len__(self):
        """Return the number of cells in the dataset."""
        return self.adata.shape[0]
