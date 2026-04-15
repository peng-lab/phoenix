""" 
Spatial transcriptomics dataset class for H5PY
© Peng Lab / Helmholtz Munich
"""

import h5py
from PIL import Image
from torch.utils.data import Dataset

#------------------------------------------------------------------------------------------


class H5PYDataset(Dataset):
    """
    The spatial transcriptomics dataset class for H5PY.
    """
    def __init__(self, image_path: str, transform=None):
        """
        initiate with image_path and transform
        """
        self.image_path = image_path
        self.transform = transform

        # open once just to get dataset sizes
        with h5py.File(self.image_path, 'r') as f:
            self.num_samples = f['patches'].shape[0]

    def __getitem__(self, idx):
        # open lazily when accessing
        with h5py.File(self.image_path, 'r') as f:
            patch = f['patches'][idx]
            coord = f['coords'][idx]

        # apply the image transform
        patch = Image.fromarray(patch)
        if self.transform:
            patch = self.transform(patch)

        return patch, coord

    def __len__(self):
        """
        :return length of dataset
        """
        return self.num_samples
