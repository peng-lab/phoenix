from setuptools import setup, find_packages

setup(
    name="phoenix",
    version="1.0.0",
    description="Phoenix - PyTorch",
    packages=find_packages(exclude=[]),
    install_requires=[
        #"apex==0.1",
        "anndata==0.12.2",
        #"flash_attn==2.7.4.post1",
        #"h5py==3.14.0",
        #"openslide-python==1.4.2",
        #"slideio==2.7.3",
        #"spatialdata==0.5.0",
        #"timm==1.0.19",
        #"torch==2.6.0",
        #"torchvision==0.21.0",
        #"transformers==4.55.4",
        #"xarray==2025.9.1",
        #"xformers==0.0.29.post3",
        "zuko==1.3.0",
    ],
)
