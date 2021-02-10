"""First-time setup.

Run `python setup.py develop` to install dependencies.
"""

from setuptools import setup, find_packages

# Setup
setup(
    name='msnn',
    description='An SNN infrastructure from the MacLean group',
    version='2.0',
    packages=find_packages(),
    install_requires=[
        'dataclasses_json',
        'hjson',
        'matplotlib',
        'tensorflow'
    ]
)
