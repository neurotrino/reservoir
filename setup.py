from setuptools import setup, find_packages

setup = setup(
    name="msnn",
    description="An infrastructure for the study of spiking neural networks",
    author="MacLean Lab",
    version="2.1.0",
    packages=find_packages(),
    install_requires=[
        "h5py >= 2.8"
        "keras-tuner >= 1.1",              # kt
        "matplotlib >= 3.5",               # plt (matplotlib.pyplot)
        "numpy >= 1.22",                   # np
        "scikit-learn >= 1.0",             # sklearn
        "scipy >= 1.8",
        "seaborn >= 0.11",                 # sns
        "tensorflow >= 2.9",               # tf
        "tensorflow_probability >= 0.16",  # tfp
    ],
    python_requires=">= 3.10",
)
