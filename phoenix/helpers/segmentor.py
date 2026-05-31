"""
Fast and simple cell nuclei segmentation
© Peng Lab / Helmholtz Munich
"""

import h5py
import numpy as np
import openslide
from instanseg import InstanSeg
from pathlib import Path
from skimage import io, measure
from tqdm import tqdm

#------------------------------------------------------------------------------------------

class NucleiPatchExtractor:
    """
    Segment nuclei in an H&E image and extract patches centered on each.
    """
    def __init__(self, native_mpp: float, target_mpp: float, patch_size: int):
        scaling_factor = target_mpp / native_mpp
        self.native_mpp = native_mpp
        self.patch_size = int(patch_size * scaling_factor)
        self.half_size = self.patch_size // 2
        self.segmentor = InstanSeg("brightfield_nuclei", verbosity=0)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Return an int32 label mask (H, W) for an RGB input image.
        """
        labeled, _ = self.segmentor.eval_small_image(image, pixel_size=self.native_mpp)
        return labeled.squeeze().cpu().numpy().astype(np.int32)

    def extract(self, image: np.ndarray, masks: np.ndarray):
        """
        Return (patches [N, P, P, 3] uint8, coords [N, 2] int32).
        """
        properties = measure.regionprops(masks)
        pad_array = np.pad(
            image, (
                (self.half_size, self.half_size),
                (self.half_size, self.half_size),
                (0, 0),
            ),
            mode="reflect"
        )
        patches = np.empty(
            (
                len(properties),
                self.patch_size,
                self.patch_size,
                3,
            ), dtype=np.uint8
        )
        coords = np.empty((len(properties), 2), dtype=np.int32)
        for i, p in tqdm(enumerate(properties)):
            cy, cx = int(round(p.centroid[0])), int(round(p.centroid[1]))
            patches[i] = pad_array[
                cy:cy + self.patch_size,
                cx:cx + self.patch_size,
            ]
            coords[i] = (cx, cy)
        return patches, coords

    def process(self, img_path: str, out_dir: str) -> tuple[str, int]:
        """
        Segment a WSI and save patches + coords to an HDF5 file.
        """
        try:
            slide = openslide.OpenSlide(img_path)
            W, H = slide.level_dimensions[0]
            image = np.array(slide.read_region((0, 0), 0, (W, H)))[..., :3]
        except Exception:
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
                )
            )
            f.create_dataset("coords", data=coords)
        print(f"Saved {len(masks)} patches!")
        return h5_path, len(patches)
