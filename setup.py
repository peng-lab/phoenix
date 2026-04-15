from setuptools import setup, find_packages

setup(
    name="phoenix",
    version="1.0.0",
    description="Phoenix - PyTorch",
    packages=find_packages(exclude=[]),
    install_requires=[
        #"timm==1.0.19",
        #"torch==2.6.0",
        #"torchvision==0.21.0",
        #"transformers==4.55.4",
        "zuko==1.3.0",
    ],
)
