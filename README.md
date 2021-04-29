# ProtoTorch Models

Pre-packaged prototype-based machine learning models using ProtoTorch and
PyTorch-Lightning.

## Installation

To install this plugin, simple install
[ProtoTorch](https://github.com/si-cim/prototorch) first by following the
installation instructions there and then install this plugin by doing:

```sh
git clone https://github.com/si-cim/prototorch_models.git && cd prototorch_models
pip install -e .
```

The plugin should then be available for use in your Python environment as
`prototorch.models`.

## Development setup

It is recommended that you use a virtual environment for development. If you do
not use `conda`, the easiest way to work with virtual environments is by using
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). Once
you've installed it with `pip install virtualenvwrapper`, you can do the
following:

```sh
export WORKON_HOME=~/pyenvs
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh  # might be different
# source ~/.local/bin/virtualenvwrapper.sh
mkvirtualenv pt
workon pt
git clone git@github.com:si-cim/prototorch_models.git
cd prototorch_models
git checkout dev
pip install -e .[all]  # \[all\] if you are using zsh or MacOS
```

To assist in the development process, you may also find it useful to install
`yapf`, `isort` and `autoflake`. You can install them easily with `pip`.

## Available models

- GLVQ
- Siamese GLVQ
- Neural Gas

## Work in Progress
- CBC

## Planned models
- GMLVQ
- Local-Matrix GMLVQ
- Limited-Rank GMLVQ
- GTLVQ
- RSLVQ
- PLVQ
- LVQMLN
