# ProtoTorch Models

[![Build Status](https://travis-ci.org/si-cim/prototorch_models.svg?branch=main)](https://travis-ci.org/si-cim/prototorch_models)
[![PyPI](https://img.shields.io/pypi/v/prototorch_models)](https://pypi.org/project/prototorch_models/)

Pre-packaged prototype-based machine learning models using ProtoTorch and
PyTorch-Lightning.

## Installation

To install this plugin, simply run the following command:

```sh
pip install prototorch_models
```

**Installing the models plugin should automatically install a suitable version
of** [ProtoTorch](https://github.com/si-cim/prototorch). The plugin should then
be available for use in your Python environment as `prototorch.models`.

## Available models

### LVQ Family

- Learning Vector Quantization 1 (LVQ1)
- Generalized Learning Vector Quantization (GLVQ)
- Generalized Relevance Learning Vector Quantization (GRLVQ)
- Generalized Matrix Learning Vector Quantization (GMLVQ)
- Localized and Generalized Matrix Learning Vector Quantization (LGMLVQ)
- Limited-Rank Matrix Learning Vector Quantization (LiRaMLVQ)
- Learning Vector Quantization Multi-Layer Network (LVQMLN)
- Siamese GLVQ
- Cross-Entropy Learning Vector Quantization (CELVQ)
- Robust Soft Learning Vector Quantization (RSLVQ)

### Other

- k-Nearest Neighbors (KNN)
- Neural Gas (NG)
- Growing Neural Gas (GNG)

## Work in Progress

- Classification-By-Components Network (CBC)
- Learning Vector Quantization 2.1 (LVQ2.1)

## Planned models

- Median-LVQ
- Generalized Tangent Learning Vector Quantization (GTLVQ)
- Probabilistic Learning Vector Quantization (PLVQ)
- Self-Incremental Learning Vector Quantization (SILVQ)

## Development setup

It is recommended that you use a virtual environment for development. If you do
not use `conda`, the easiest way to work with virtual environments is by using
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). Once
you've installed it with `pip install virtualenvwrapper`, you can do the
following:

```sh
export WORKON_HOME=~/pyenvs
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh  # location may vary
mkvirtualenv pt
```

Once you have a virtual environment setup, you can start install the `models`
plugin with:

```sh
workon pt
git clone git@github.com:si-cim/prototorch_models.git
cd prototorch_models
git checkout dev
pip install -e .[all]  # \[all\] if you are using zsh or MacOS
```

To assist in the development process, you may also find it useful to install
`yapf`, `isort` and `autoflake`. You can install them easily with `pip`. **Also,
please avoid installing Tensorflow in this environment. It is known to cause
problems with PyTorch-Lightning.**

## FAQ

### How do I update the plugin?

If you have already cloned and installed `prototorch` and the
`prototorch_models` plugin with the `-e` flag via `pip`, all you have to do is
navigate to those folders from your terminal and do `git pull` to update.
