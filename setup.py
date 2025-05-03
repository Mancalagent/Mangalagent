from setuptools import setup, find_packages


import os

# Utility to read requirements.txt
def parse_requirements(filename='requirements.txt'):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename)) as f:
        lines = f.read().splitlines()
    # Filter out comments, empty lines, editable installs, etc.
    reqs = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith('#')
        and not line.startswith('-e')
    ]
    return reqs


setup(
    name="mangalagent",
    version="0.1.0",
    packages=find_packages(),         # <-- picks up every folder with __init__.py
    python_requires=">=3.9",
    install_requires=parse_requirements(),
)