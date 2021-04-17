"""First-time setup.

Run `python setup.py develop` to install dependencies.
"""

from setuptools import setup, find_packages

import os
import subprocess


def installation_wrapper(setup):
    """Code to be run when the setup script is invoked (i.e. when we
    call `pip install -e .`).
    """

    # Detect environment
    try:
        # Conda
        venv_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    except:
        # Standard venv
        venv_path = os.path.join(os.environ['VIRTUAL_ENV'], 'lib')

    # Find the directory we need to create our .pth file in
    for path, dirs, _ in os.walk(venv_path):
        if 'site-packages' in dirs:
            venv_path = os.path.join(path, 'site-packages')

    # Bake gitsha into the venv
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    sha = sha.decode('utf-8')  # returned as bytes, so we decode

    # Extensible
    evars = [  # (name, value)
        ('GITSHA', sha),
    ]

    # Creat environment variables
    with open(os.path.join(venv_path, 'msnn-custom-evars.pth'), 'w+') as file:

        pth_txt = "import os;"

        for (k, v) in evars:
            pth_txt += f"os.environ['MSNN_{k}']='{v}';"

        file.write(pth_txt)

    return setup


# Setup
setup = installation_wrapper(setup(
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
))
