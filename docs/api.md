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
architecture built on `apex`, `flash-attn`, and `xformers`. `flash-attn` and `xformers` have
prebuilt wheels (see README.md's "Optimized model variant" section) but are platform-specific
(linux x86_64, sm_80+ GPU) and are not part of the docs build environment; `apex` has no wheel
index at all and must be built from source against the installed torch. The modules cannot be
imported here and are intentionally left out of the generated API reference; see their
docstrings in the source for usage.

## Trainers

```{eval-rst}
.. module:: phoenix.trainers
.. currentmodule:: phoenix

.. autosummary::
    :toctree: generated

    trainers.mixer_trainer.MixerTrainer
    trainers.mixer_trainer.TrainerConfig
    trainers.mixer_trainer.WarmupCosineAnnealingLR
    trainers.mixer_trainer.move_to
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
    helpers.demo_plot.to_dense
    helpers.demo_plot.norm01
    helpers.demo_plot.set_active_layer
    helpers.segmentor.NucleiPatchExtractor
```
