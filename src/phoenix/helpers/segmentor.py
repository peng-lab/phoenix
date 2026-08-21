"""
Fast and simple cell nuclei segmentation.

© Peng Lab / Helmholtz Munich
"""

from pathlib import Path

import h5py
import numpy as np
import openslide
from instanseg import InstanSeg
from skimage import io, measure
from tqdm import tqdm

# ------------------------------------------------------------------------------------------


class NucleiPatchExtractor:
    """
    Segment nuclei in an H&E image and extract patches centered on each.

    Parameters
    ----------
    native_mpp
        Native resolution of the input image, in microns per pixel.
    target_mpp
        Target resolution, in microns per pixel; together with `native_mpp`
        determines the extracted patch size in pixels.
    patch_size
        Patch side length, in pixels, at `target_mpp`.
    """

    def __init__(self, native_mpp: float, target_mpp: float, patch_size: int):
        scaling_factor = target_mpp / native_mpp
        self.native_mpp = native_mpp
        self.patch_size = int(patch_size * scaling_factor)
        self.half_size = self.patch_size // 2
        self.segmentor = InstanSeg("brightfield_nuclei", verbosity=0)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment nuclei in an RGB image.

        Parameters
        ----------
        image
            RGB input image.

        Returns
        -------
        Int32 label mask of shape ``(H, W)``.
        """
        labeled, _ = self.segmentor.eval_small_image(image, pixel_size=self.native_mpp)
        return labeled.squeeze().cpu().numpy().astype(np.int32)

    def extract(self, image: np.ndarray, masks: np.ndarray):
        """
        Extract a fixed-size patch centered on each labeled nucleus.

        Parameters
        ----------
        image
            RGB image the label mask was computed from.
        masks
            Int label mask of shape ``(H, W)``, as returned by `segment`.

        Returns
        -------
        Tuple of the extracted patches, shape ``(N, patch_size, patch_size, 3)`` uint8,
        and their center coordinates, shape ``(N, 2)`` int32.
        """
        properties = measure.regionprops(masks)
        pad_array = np.pad(
            image,
            (
                (self.half_size, self.half_size),
                (self.half_size, self.half_size),
                (0, 0),
            ),
            mode="reflect",
        )
        patches = np.empty(
            (
                len(properties),
                self.patch_size,
                self.patch_size,
                3,
            ),
            dtype=np.uint8,
        )
        coords = np.empty((len(properties), 2), dtype=np.int32)
        for i, p in tqdm(enumerate(properties)):
            cy, cx = int(round(p.centroid[0])), int(round(p.centroid[1]))
            patches[i] = pad_array[
                cy : cy + self.patch_size,
                cx : cx + self.patch_size,
            ]
            coords[i] = (cx, cy)
        return patches, coords

    def process(self, img_path: str, out_dir: str) -> tuple[str, int]:
        """
        Segment a whole-slide image and save its patches and coordinates to HDF5.

        Reads `img_path` via OpenSlide when possible, falling back to a plain image
        read (dropping any alpha channel) otherwise.

        Parameters
        ----------
        img_path
            Path to the whole-slide (or plain) input image.
        out_dir
            Directory the output HDF5 file is written to; the filename is derived
            from `img_path`'s stem.

        Returns
        -------
        Tuple of the output HDF5 path and the number of extracted patches.
        """
        try:
            slide = openslide.OpenSlide(img_path)
            W, H = slide.level_dimensions[0]
            image = np.array(slide.read_region((0, 0), 0, (W, H)))[..., :3]
        except Exception:  # noqa: BLE001 -- OpenSlide raises many undocumented, format-specific
            # errors; any failure here is intentionally treated as "not a whole-slide image"
            # and falls back to a plain image read.
            image = io.imread(img_path)
            if image.shape[-1] == 4:
                image = image[..., :3]
        masks = self.segment(image)
        patches, coords = self.extract(image, masks)

        h5_path = f"{out_dir}/{Path(img_path).stem}.h5"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "patches",
                data=patches,
                compression="gzip",
                compression_opts=4,
                chunks=(
                    1,
                    self.patch_size,
                    self.patch_size,
                    3,
                ),
            )
            f.create_dataset("coords", data=coords)
        print(f"Saved {len(masks)} patches!")
        return h5_path, len(patches)
