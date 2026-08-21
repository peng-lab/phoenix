import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("PIL")

from phoenix.datasets.h5py_dataset import H5PYDataset  # noqa: E402


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


def test_spatial_dataset_is_a_torch_dataset():
    # Instantiating SpatialDataset needs a real SpatialData .zarr store (he_image +
    # nucleus_boundaries elements, a table with a matching gene panel), which isn't
    # practical to fabricate in a unit test; this at least confirms the class is
    # importable and conforms to the expected torch Dataset interface.
    pytest.importorskip("spatialdata")
    from torch.utils.data import Dataset

    from phoenix.datasets.zarr_dataset import SpatialDataset

    assert issubclass(SpatialDataset, Dataset)
    assert hasattr(SpatialDataset, "__getitem__")
    assert hasattr(SpatialDataset, "__len__")
