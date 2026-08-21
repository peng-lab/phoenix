"""
Spatial transcriptomics dataset based on H5PY
© Peng Lab / Helmholtz Munich
"""

import h5py
from PIL import Image
from torch.utils.data import Dataset

# ------------------------------------------------------------------------------------------


class H5PYDataset(Dataset):
    """
    Spatial transcriptomics dataset backed by an HDF5 file of image patches and coordinates.

    Opens the underlying HDF5 file lazily on each item access, keeping only the dataset
    length cached from an initial read at construction time.

    Parameters
    ----------
    image_path
        Path to an HDF5 file containing a ``patches`` dataset (image patches) and a
        ``coords`` dataset (their spatial coordinates), both indexed along the first axis.
    transform
        Optional torchvision-style transform applied to each patch after it is loaded.
    """

    def __init__(self, image_path: str, transform=None):
        self.image_path = image_path
        self.transform = transform

        # open once just to get dataset sizes
        with h5py.File(self.image_path, "r") as f:
            self.num_samples = f["patches"].shape[0]

    def __getitem__(self, idx):
        """
        Load and transform a single patch/coordinate pair.

        Parameters
        ----------
        idx
            Index into the underlying HDF5 datasets.

        Returns
        -------
        Tuple of the (optionally transformed) image patch and its coordinate array.
        """
        # open lazily when accessing
        with h5py.File(self.image_path, "r") as f:
            patch = f["patches"][idx]
            coord = f["coords"][idx]

        # apply the image transform
        patch = Image.fromarray(patch)
        if self.transform:
            patch = self.transform(patch)

        return patch, coord

    def __len__(self):
        """
        Return the number of patches in the dataset.
        """
        return self.num_samples
