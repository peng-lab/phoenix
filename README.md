# Phoenix 🐦‍🔥
### Phoenix reveals the spatial mechanisms of cancer de novo from routine histology

[[preprint](https://www.medrxiv.org/content/10.1101/2024.03.15.24304211v1)] [[weights](https://huggingface.co/peng-lab/phoenix)] [[notebook](https://github.com/marrlab/HistoGPT/blob/main/tutorial-2.ipynb)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marrlab/HistoGPT/blob/main/tutorial-2.ipynb)

Phoenix is a (latent) flow matching generative model that predicts spatially resolved single-cell gene expression directly from routine H&E-stained histology images. It generalizes across cohorts, donors, organs, and tissues — enabling in silico analysis of treatment response and tissue organization at population scale.

<img src="github/Figure-1.png" width="800"/>

### Phoenix is simple and easy to use

We can install Phoenix with the following commands
```
pip install git+https://github.com/peng-lab/phoenix
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
