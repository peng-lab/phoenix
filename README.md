# Phoenix 🐦‍🔥
### Phoenix reveals the spatial mechanisms of cancer de novo from routine histology

[[preprint](https://www.medrxiv.org/content/10.1101/2024.03.15.24304211v1)] [[weights](https://huggingface.co/peng-lab/phoenix)] [[notebook](https://github.com/marrlab/HistoGPT/blob/main/tutorial-2.ipynb)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marrlab/HistoGPT/blob/main/tutorial-2.ipynb)

Phoenix is a (latent) flow matching generative model that predicts spatially resolved single-cell gene expression directly from routine H&E-stained histology images. It generalizes across cohorts, donors, organs, and tissues — enabling in silico analysis of treatment response and tissue organization at population scale.

<img src="github/Figure-1.png" width="800"/>

### Phoenix is simple and easy to use

We can install Phoenix with the following command
```
pip install git+https://github.com/peng-lab/phoenix
```

We now load the gene list, normalization statistics, and image patches
```python
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from github.datasets.h5py_dataset import H5PYDataset

gene_path = './xenium_human_multi.npy'
gene_list = list(np.load(gene_path))

stats_path = "./tenx-multi-table.npz"
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

To load the vision encoder and flow transformer use
```python
from github.models.flow_simple import FlowTransformerModel, FlowTransformerConfig

vision_model = timm.create_model(
    "hf-hub:bioptimus/H-optimus-1",
    pretrained=True,
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

To predict gene expression from histology image use
```python
from github.helpers.inference import FlowPipeline

pipeline = FlowPipeline(
    model=flow_model,
    stats=statistics,
    t_0=0.0,
    t_1=1.0,
    atol=1e-0,
    rtol=1e-0,
)

gex_pred, coords_list = pipeline(gene_list, dataloader)
```
