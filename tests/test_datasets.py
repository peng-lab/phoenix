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

from .conftest import make_synthetic_store  # noqa: E402


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


# ------------------------------------------------------------------------------------------
# field of view: a patch must cover 224 px * 0.5 micron = 112 micron whatever the store layout

FOV_UM = 112.0
IMAGE_SIZE = 1536
XENIUM_SOURCE_MPP = 0.2125  # morphology pixel size, the xenium reader's global unit
XENIUM_HE_MPP = 0.2737
HE_ONLY_HE_MPP = 0.221


def make_layout_store(path, layout, attrs="complete", **kwargs):
    """
    Write a store laid out like a real xenium or h&e-only one, and return its true he pixel size.

    ``xenium``: global is the morphology pixel grid, shapes are in micron
    (``Scale(1 / source_mpp)``) and the he image carries a 90 degree rotation scaled by
    ``he_mpp / source_mpp``. ``he``: global is the he pixel grid and both elements are
    identity-transformed.
    """
    from spatialdata.transformations import Affine, Scale

    if layout == "xenium":
        s = XENIUM_HE_MPP / XENIUM_SOURCE_MPP
        he_to_global = Affine(
            np.array([[0, s, 0], [-s, 0, s * IMAGE_SIZE], [0, 0, 1]]),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        shape_to_global = Scale([1 / XENIUM_SOURCE_MPP] * 2, axes=("x", "y"))
        full = {
            "spatialdata_io_reader": "xenium",
            "source_mpp": XENIUM_SOURCE_MPP,
            "source_he_mpp": XENIUM_HE_MPP,
        }
        he_mpp, margin = XENIUM_HE_MPP, 260
    else:
        he_to_global = shape_to_global = None
        full = {"spatialdata_io_reader": "he", "source_mpp": XENIUM_SOURCE_MPP, "source_he_mpp": HE_ONLY_HE_MPP}
        he_mpp, margin = HE_ONLY_HE_MPP, 300
    store = make_synthetic_store(
        path,
        image_size=IMAGE_SIZE,
        margin=margin,
        attrs=full if attrs == "complete" else attrs,
        he_to_global=he_to_global,
        shape_to_global=shape_to_global,
        **kwargs,
    )
    return store, he_mpp


@pytest.mark.parametrize("layout", ["xenium", "he"])
def test_native_mpp_is_the_he_pixel_size(tmp_path, layout):
    store, he_mpp = make_layout_store(tmp_path / "store.zarr", layout)
    dataset = SpatialDataset(store, "table", GENES)
    assert dataset.native_mpp == pytest.approx(he_mpp, rel=1e-6)


@pytest.mark.parametrize("layout", ["xenium", "he"])
def test_patch_covers_112_micron(tmp_path, layout):
    store, he_mpp = make_layout_store(tmp_path / "store.zarr", layout)
    dataset = SpatialDataset(store, "table", GENES, image_transform=DEMO_TRANSFORM)
    patch = dataset.get_patch(*(int(v) for v in dataset.adata.obsm["spatial"][0]))
    assert patch.shape[0] == patch.shape[1]
    assert abs(patch.shape[0] * he_mpp - FOV_UM) <= 2 * he_mpp
    image, _ = dataset[0]
    assert image.shape == (3, 224, 224)


@pytest.mark.parametrize("layout", ["xenium", "he"])
def test_patch_is_centred_on_the_cell(tmp_path, layout):
    store, _ = make_layout_store(tmp_path / "store.zarr", layout, blob_cell=0)
    dataset = SpatialDataset(store, "table", GENES)
    patch = dataset.get_patch(*(int(v) for v in dataset.adata.obsm["spatial"][0]))
    ys, xs = np.nonzero(patch[..., 0])
    centre = patch.shape[0] / 2
    assert abs(ys.mean() - centre) <= 1.5
    assert abs(xs.mean() - centre) <= 1.5


def test_native_mpp_without_reader_attr_uses_shape_scale(tmp_path):
    # stores without attrs (e.g. the demo store) keep deriving the pixel size from the transforms
    store, he_mpp = make_layout_store(tmp_path / "store.zarr", "xenium", attrs={})
    assert SpatialDataset(store, "table", GENES).native_mpp == pytest.approx(he_mpp, rel=1e-6)


def test_native_mpp_override_wins(tmp_path):
    store, _ = make_layout_store(tmp_path / "store.zarr", "he")
    assert SpatialDataset(store, "table", GENES, native_mpp=0.3).native_mpp == 0.3


@pytest.mark.parametrize(
    ("layout", "attrs"),
    [
        ("he", {}),  # identity transforms and no attrs: the pixel size is unknowable
        ("he", {"spatialdata_io_reader": "he"}),  # missing source_he_mpp
        ("xenium", {"spatialdata_io_reader": "xenium"}),  # missing source_mpp
    ],
)
def test_native_mpp_raises_when_unknown(tmp_path, layout, attrs):
    store, _ = make_layout_store(tmp_path / "store.zarr", layout, attrs=attrs)
    with pytest.raises(ValueError, match="mpp"):
        SpatialDataset(store, "table", GENES)
