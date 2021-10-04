from setuptools import setup, find_packages

setup = setup(
    name='msnn',
    description='An SNN infrastructure from the MacLean group',
    version='2.1.0',
    packages=find_packages(),
    install_requires=[
        'dataclasses_json',
        'hjson',
        'matplotlib',
        'tensorflow',
        'tensorflow-probability'
    ]
)
