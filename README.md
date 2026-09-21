# Phoenix 🐦‍🔥
### Pan-cancer virtual spatial transcriptomics from routine histology with Phoenix

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[[preprint](https://doi.org/10.64898/2026.04.25.720812)] [[weights](https://huggingface.co/peng-lab/phoenix)] [[notebook](https://github.com/peng-lab/phoenix/blob/main/phoenix_demo.ipynb)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/peng-lab/phoenix/blob/main/phoenix_demo.ipynb)

Phoenix is a (latent) flow matching generative model that predicts spatially resolved single-cell gene expression directly from routine H&E-stained histology images. It generalizes across cohorts, donors, organs, and tissues — enabling in silico analysis of tissue organization and treatment response at population scale.

<img src="https://raw.githubusercontent.com/peng-lab/phoenix/main/docs/_static/img/figure-1.jpg" width="800"/>

<br>

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].

Install the latest development version:

```bash
pip install git+https://github.com/peng-lab/phoenix.git  # (or `uv add`)
```

### Optimized model variant (H100 / CUDA 13)

`phoenix.models.flow_llama3` and `phoenix.models.mlp_mixer_ae` are optimized reimplementations of
the flow transformer and the latent autoencoder, built on `apex`, `flash-attn` and `xformers` fused
kernels. `phoenix.models.flow_simple` is a pure-torch equivalent that needs none of this and is what
the `full` extra alone gives you — use it unless you specifically need the optimized variant's speed.

Requirements: Linux x86_64, Python 3.12–3.14, a CUDA 13.0 toolkit (`nvcc`) to build `apex`, and a GPU
of compute capability sm_80 or newer (A100/H100/H200) — the published `flash-attn` wheels carry no
kernels for older GPUs (e.g. sm_75), so this stack imports but cannot run on those.

```bash
git clone https://github.com/peng-lab/phoenix.git && cd phoenix
uv sync --extra full          # installs torch==2.12.1 / torchvision==0.27.1, pinned to match
                               # the flash-attn build below -- there is no CUDA-13 flash-attn
                               # wheel for torch 2.13 yet

uv pip install --index https://wheels.astral.sh/simple/cu130/ \
    "flash-attn==2.8.3.post1+cu.13.0.torch.2.12"

uv pip install "xformers>=0.0.35"
```

`apex` has no wheel index and must be built from source against the torch just installed, since its
`fused_layer_norm_cuda` extension (used by `apex.normalization.FusedRMSNorm`) links torch's ABI:

```bash
git clone https://github.com/NVIDIA/apex.git && cd apex
uv pip install ninja

# TORCH_CUDA_ARCH_LIST must match your target GPU's compute capability (9.0 for H100). Leaving it
# unset lets torch auto-detect the *build* host's GPU, which silently produces a wheel that only
# runs there.
TORCH_CUDA_ARCH_LIST="9.0" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv build --wheel --no-build-isolation -o /path/to/wheel/output .

uv pip install /path/to/wheel/output/apex-0.1-*.whl
```

On a SLURM cluster, run the build step above as a batch job submitted from a node with the target
GPU (compute nodes commonly have no internet access, so `git clone` and `uv pip install ninja`
above must run on a login/head node first).

> [!WARNING]
> `uv sync` removes packages that aren't declared in `pyproject.toml`. Once `flash-attn`, `xformers`
> and `apex` are installed on top of `uv sync --extra full`, every *later* sync must add `--inexact`
> (`uv sync --extra full --inexact`), or it will silently uninstall all three again.

Verify the install:

```bash
python -c "
import torch, flash_attn, xformers
from apex.normalization import FusedRMSNorm
from flash_attn import flash_attn_func
from xformers.ops import SwiGLU
print(torch.__version__, torch.version.cuda)
"
```

`flash-attn`'s kernels only run on an sm_80+ device, so the import above succeeds anywhere but the
forward pass itself must be run on the target GPU.

### Usage

To load the 224x224 patches saved in an H5 file use
```python
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from phoenix.datasets.h5py_dataset import H5PYDataset

gene_path = './xenium_human_multi.npy'
gene_list = list(np.load(gene_path))

stats_path = "./stats_table.npz"
statistics = np.load(stats_path)

bicubic = InterpolationMode.BICUBIC
image_transform = v2.Compose(
    [
        v2.Resize((224, 224), bicubic),
        v2.CenterCrop((224, 224)),
        v2.ToTensor(),
        v2.Normalize(
            (0.707223, 0.578729, 0.703617),
            (0.211883, 0.230117, 0.177517),
        ),
    ]
)

image_path = "./demo_patch.h5"
dataset = H5PYDataset(
    image_path=image_path,
    transform=image_transform,
)
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
print('Length dataset & dataloader:', (len(dataset), len(dataloader)))
```

To load the model weights hosted on HuggingFace use
<br>
(We recommend using the model trained on the Nest)
```
#https://huggingface.co/peng-lab/phoenix/resolve/main/weights/flow/tenx/multi/cell/20x/discrete/flow_model.pth
https://huggingface.co/peng-lab/phoenix/resolve/main/weights/flow/nest/multi/cell/20x/discrete/flow_model.pth
```

To load the vision encoder and flow transformer use
<br>
(We recommend using the optimized implementation)
```python
from phoenix.models.flow_llama3 import FlowTransformerModel, FlowTransformerConfig
#from phoenix.models.flow_simple import FlowTransformerModel, FlowTransformerConfig

vision_model = timm.create_model(
    "vit_giant_patch14_reg4_dinov2",
    pretrained=False,
    img_size=224,
    num_classes=0,
    global_pool="token",
    init_values=1e-5,
    dynamic_img_size=False,
)

flow_model = FlowTransformerModel(
    FlowTransformerConfig(
        d_genes=1,
        d_image=1536,
        d_model=512,
        d_cross=512,
        n_heads=8,
        n_layers=8,
        qkv_bias=False,
        ffn_bias=False,
        ffn_mult=4,
        attn_drop=0.0,
        proj_drop=0.0,
        n_classes=0,
        cls_drop=0.1,
        checkpoint=False,
    ),
    vision_model=vision_model
)

state_dict = torch.load(state_path, map_location='cuda:0')
flow_model.load_state_dict(state_dict, strict=True)
flow_model = flow_model.eval().cuda()
```

To make a forward pass and check that it works use
```python
x = torch.rand(1, 377, 1).cuda()
t = torch.rand(x.shape[0]).cuda()
c = torch.rand(1, 256, 1536).cuda()

output = flow_model(x, t, c)
print("Output:", output.size())
```

To predict gene expression from histology images use
```python
from phoenix.helpers.inference import FlowPipeline

pipeline = FlowPipeline(
    model=flow_model,
    stats=statistics,
    t_0=0.0,
    t_1=1.0,
    atol=1e-1,
    rtol=1e-1,
)

gex_pred, coords_list = pipeline(gene_list, dataloader)
```

## Release notes

See the [changelog][].

## Contact

For questions, help requests, and bug reports, please use the [issue tracker][].

## Citation

In case you found our work useful, please consider citing us:
```
@article{tran/gindra2026.04.25.720812,
	author = {Tran, Manuel and Gindra, Rushin H. and Putze, Philipp and Senbai, Kang and Palla, Giovanni and Kos, Tina and Falcomat{\`a}, Chiara and Wang, Chen and Guo, Ruifeng (Ray) and Boxberg, Melanie and Berclaz, Luc M. and Lindner, Lars H. and Bergmayr, Linda and Kn{\"o}sel, Thomas and Jurmeister, Philipp and Klauschen, Frederick and Homicsko, Krisztian and Gottardo, Raphael and Eckstein, Markus and Matek, Christian and Mock, Andreas and Theis, Fabian J. and Saur, Dieter and Peng, Tingying},
	title = {Pan-cancer virtual spatial transcriptomics from routine histology with Phoenix},
	year = {2026},
	journal = {bioRxiv},
	doi = {https://doi.org/10.64898/2026.04.25.720812},
}
```

## License

Phoenix is released under the [PolyForm Noncommercial License 1.0.0][license].

Any noncommercial purpose is permitted, including academic research, personal study, and
hobby projects. Commercial use requires a separate license — please get in touch.
See the [full license text][license] for the exact terms.

[uv]: https://github.com/astral-sh/uv
[license]: https://github.com/peng-lab/phoenix/blob/main/LICENSE
[issue tracker]: https://github.com/peng-lab/phoenix/issues
[tests]: https://github.com/peng-lab/phoenix/actions/workflows/test.yaml
[badge-tests]: https://img.shields.io/github/actions/workflow/status/peng-lab/phoenix/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/phoenix/badge/
[documentation]: https://phoenix.readthedocs.io
[changelog]: https://phoenix.readthedocs.io/page/changelog.html
