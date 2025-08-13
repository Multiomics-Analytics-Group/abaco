## ![ABaCo Logo](https://raw.githubusercontent.com/Multiomics-Analytics-Group/abaco/HEAD/docs/images/logo/abaco_logo.png)

<p align="center"><em>Batch Effect Correction framework for metagenomic data</em></p>

<p align="center">
    <a href="https://pypi.org/project/abaco/">
        <img src="https://img.shields.io/pypi/v/abaco?label=PyPI" alt="PyPI">
    </a>
    <a href="https://github.com/Multiomics-Analytics-Group/abaco/actions/workflows/cicd.yml">
        <img src="https://github.com/Multiomics-Analytics-Group/abaco/actions/workflows/cicd.yml/badge.svg?branch=" alt="Python application">
    </a>
    <a href="https://abaco.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/abaco/badge/?version=latest" alt="Read the Docs">
    </a>
    <img src="https://img.shields.io/pypi/pyversions/abaco" alt="PyPI - Python Version">
    <br>
    <br>
    <img src="https://img.shields.io/github/issues/Multiomics-Analytics-Group/abaco" alt="GitHub issues">
    <img src="https://img.shields.io/github/license/Multiomics-Analytics-Group/abaco" alt="GitHub license">
    <img src="https://img.shields.io/github/last-commit/Multiomics-Analytics-Group/abaco" alt="GitHub last commit">
    <img src="https://img.shields.io/github/stars/Multiomics-Analytics-Group/abaco?style=social" alt="GitHub stars">
</p>

The integration of metagenomic data from multiple studies and experimental conditions is essential to understand the interactions between microbial communities in complex biological systems, but the inherent diversity and biological complexity pose methodological challenges that require refined strategies for atlas-level integration. ABaCo, a family of generative models based on Variational Autoencoders (VAEs) combined with an adversarial training, aim for the integration of metagenomic data from different studies by minimizing technical heterogeneity conserving biological significance. The VAE encodes the data into a latent space, while the discriminator is trained to detect the provenance of the data, eliminating variability associated with its origin; concurrently, the data is modeled using distributions suitable for raw counts, and the latent space follows a clustering prior to ensure biological conservation.

An overview of the ABaCo workflow is shown in the figure below:

![ABaCo Abstract](https://raw.githubusercontent.com/Multiomics-Analytics-Group/abaco/HEAD/docs/images/abaco_overview.png)

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Installation 

ABaCo is available on PyPI: 
```bash
pip install abaco
```

## Features

## Usage

## Documentation
Tutorials and documentation are available on [Read the Docs](https://mona-abaco.readthedocs.io/)

## Contributing
1. Fork the repository
2. Clone the repository
3. Create a virtual env e.g.
  ```bash
  # navigate terminal to repo
  cd <path-to-repo-root>

  # create virtual env
  python -m venv .venv

  # activate venv
  source .venv/bin/activate
  ```
4. Install abaco in editing mode into the virtual env
  ```bash
  pip install -e .
  ```
5. Make changes
  > Note: we aimt o use numpy style python docstrings [sphinx example](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy)
6. Make a pull request

## License
