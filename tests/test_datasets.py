import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("PIL")
sd = pytest.importorskip("spatialdata")

import torch  # noqa: E402
from torchvision.transforms import InterpolationMode, v2  # noqa: E402

from phoenix.datasets.h5py_dataset import H5PYDataset  # noqa: E402
from phoenix.datasets.zarr_dataset import SpatialDataset  # noqa: E402


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "patches.h5"
    patches = (np.random.rand(4, 8, 8, 3) * 255).astype(np.uint8)
    coords = np.arange(8).reshape(4, 2).astype(np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("patches", data=patches)
        f.create_dataset("coords", data=coords)
    return path


def test_h5py_dataset_length(h5_path):
    dataset = H5PYDataset(image_path=str(h5_path))
    assert len(dataset) == 4


def test_h5py_dataset_getitem_without_transform(h5_path):
    dataset = H5PYDataset(image_path=str(h5_path))
    patch, coord = dataset[0]
    assert patch.size == (8, 8)  # PIL Image (width, height)
    np.testing.assert_array_equal(coord, [0, 1])


def test_h5py_dataset_getitem_applies_transform(h5_path):
    calls = []

    def transform(img):
        calls.append(img.size)
        return np.asarray(img)

    dataset = H5PYDataset(image_path=str(h5_path), transform=transform)
    patch, _ = dataset[2]
    assert calls == [(8, 8)]
    assert isinstance(patch, np.ndarray)


GENES = ["PECAM1", "MMRN2", "MYH11", "SFRP2"]

# the demo notebook's image transform
DEMO_TRANSFORM = v2.Compose(
    [
        v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((224, 224)),
        v2.ToTensor(),
        v2.Normalize((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)),
    ]
)


def test_spatial_dataset_length(synthetic_store):
    dataset = SpatialDataset(synthetic_store, "table", GENES)
    assert len(dataset) == dataset.sdata["table"].n_obs == 32


def test_spatial_dataset_getitem_applies_transform(synthetic_store):
    dataset = SpatialDataset(synthetic_store, "table", GENES, image_transform=DEMO_TRANSFORM)
    image, coords = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    np.testing.assert_array_equal(coords, dataset.adata.obsm["spatial"][0].astype(int))


def test_spatial_dataset_border_cell_yields_blank_patch(synthetic_store):
    dataset = SpatialDataset(synthetic_store, "table", GENES)
    image, _ = dataset[len(dataset) - 1]
    image = np.asarray(image)
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8
    assert image.max() == 0


def test_spatial_dataset_accepts_spatialdata_object(synthetic_store):
    sdata = sd.read_zarr(synthetic_store, selection=("images", "shapes", "tables"))
    from_path = SpatialDataset(synthetic_store, "table", GENES, image_transform=DEMO_TRANSFORM)
    from_sdata = SpatialDataset(sdata, "table", GENES, image_transform=DEMO_TRANSFORM)
    assert from_sdata.sdata is sdata
    assert len(from_sdata) == len(from_path)

    image_path, coords_path = from_path[0]
    image_sdata, coords_sdata = from_sdata[0]
    np.testing.assert_array_equal(coords_sdata, coords_path)
    assert torch.equal(image_sdata, image_path)


def test_spatial_dataset_subsets_to_gene_list(synthetic_store):
    dataset = SpatialDataset(synthetic_store, "table", ["MYH11", "PECAM1"])
    assert dataset.adata.var_names.tolist() == ["MYH11", "PECAM1"]
    assert dataset.gene_matrix.shape == (len(dataset), 2)
