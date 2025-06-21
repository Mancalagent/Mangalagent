from setuptools import setup, find_packages

# Directly specify the required packages
INSTALL_REQUIRES = [
    'torch',
    'numpy',
    'matplotlib',
]

setup(
    name="mangalagent",
    version="0.1.0",
    packages=find_packages(),  # picks up every folder with __init__.py
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
)