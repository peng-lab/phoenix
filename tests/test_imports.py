"""Every submodule should at least be importable given its documented extras."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name, extra_deps",
    [
        ("phoenix.datasets.h5py_dataset", ["h5py"]),
        ("phoenix.datasets.zarr_dataset", ["spatialdata"]),
        ("phoenix.helpers.demo_plot", []),
        ("phoenix.helpers.inference", []),
        ("phoenix.models.flow_simple", []),
        ("phoenix.trainers.mixer_trainer", ["pytorch_lightning"]),
    ],
)
def test_importable_with_full_extra(module_name, extra_deps):
    """These modules only need base deps + the `full` extra (torch, h5py, spatialdata, ...)."""
    pytest.importorskip("torch")
    for dep in extra_deps:
        pytest.importorskip(dep)
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name, required",
    [
        ("phoenix.models.flow_llama3", ["apex", "flash_attn", "xformers"]),
        ("phoenix.models.mlp_mixer_ae", ["apex", "xformers", "vector_quantize_pytorch"]),
        ("phoenix.helpers.segmentor", ["openslide", "instanseg", "skimage"]),
    ],
)
def test_importable_with_optional_extras(module_name, required):
    """
    These modules need packages that are deliberately not part of any pip extra
    (apex/flash-attn/xformers) or belong to the `segmentation` extra; skipped
    whenever they aren't present, which is the expected default CI environment.
    """
    for dep in required:
        pytest.importorskip(dep)
    importlib.import_module(module_name)
