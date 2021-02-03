"""First-time setup.

Run `python setup.py develop` to install dependencies.

Pretty sure this isn't the intended usage of `setup.py`, but I mean...
eh.
"""

from setuptools import setup, find_packages

# Setup
setup(
    name='maclean-snn',
    description='First SNN infrastructure from the MacLean group',
    version='2.0',
    packages=find_packages(),
    install_requires=[
        'dataclasses_json',
        'hjson',
        'matplotlib',
        'tensorflow'
    ]
)
