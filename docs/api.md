# API

## Datasets

```{eval-rst}
.. module:: phoenix.datasets
.. currentmodule:: phoenix

.. autosummary::
    :toctree: generated

    datasets.h5py_dataset.H5PYDataset
    datasets.zarr_dataset.SpatialDataset
```

## Models

```{eval-rst}
.. module:: phoenix.models
.. currentmodule:: phoenix

.. autosummary::
    :toctree: generated

    models.flow_simple.FlowTransformerModel
    models.flow_simple.FlowTransformerConfig
```

`models.flow_llama3` and `models.mlp_mixer_ae` provide optimized variants of the same
architecture built on `apex`, `flash-attn`, and `xformers`. Those packages require a matching
CUDA toolchain and are not pip-installable, so the modules cannot be imported in the docs build
environment and are intentionally left out of the generated API reference here; see their
docstrings in the source for usage.

## Trainers

```{eval-rst}
.. module:: phoenix.trainers
.. currentmodule:: phoenix

.. autosummary::
    :toctree: generated

    trainers.mixer_trainer.MixerTrainer
    trainers.mixer_trainer.TrainerConfig
```

## Helpers

```{eval-rst}
.. module:: phoenix.helpers
.. currentmodule:: phoenix

.. autosummary::
    :toctree: generated

    helpers.inference.FlowPipeline
    helpers.inference.run_flow
    helpers.demo_plot.spatial_plot
    helpers.demo_plot.spatial_zarr_plot
    helpers.demo_plot.plot_gene
```

`helpers.segmentor.NucleiPatchExtractor` depends on `openslide-python` and `instanseg` (the
`segmentation` extra), which are not yet installed in the docs build environment; it is omitted
from the generated API reference for the same reason as above.
