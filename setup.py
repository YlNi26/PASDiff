from setuptools import setup, find_packages

setup(
    name="pasdiff", 
    packages=find_packages(),
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
