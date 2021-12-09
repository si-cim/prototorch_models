"""

 ######
 #     # #####   ####  #####  ####  #####  ####  #####   ####  #    #
 #     # #    # #    #   #   #    #   #   #    # #    # #    # #    #
 ######  #    # #    #   #   #    #   #   #    # #    # #      ######
 #       #####  #    #   #   #    #   #   #    # #####  #      #    #
 #       #   #  #    #   #   #    #   #   #    # #   #  #    # #    #
 #       #    #  ####    #    ####    #    ####  #    #  ####  #    #Plugin

ProtoTorch models Plugin Package
"""
from pkg_resources import safe_name
from setuptools import find_namespace_packages, setup

PLUGIN_NAME = "models"

PROJECT_URL = "https://github.com/si-cim/prototorch_models"
DOWNLOAD_URL = "https://github.com/si-cim/prototorch_models.git"

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "prototorch>=0.7.0",
    "pytorch_lightning>=1.3.5",
    "torchmetrics",
]
CLI = [
    "jsonargparse",
]
DEV = [
    "bumpversion",
    "pre-commit",
]
DOCS = [
    "recommonmark",
    "sphinx",
    "nbsphinx",
    "ipykernel",
    "sphinx_rtd_theme",
    "sphinxcontrib-katex",
    "sphinxcontrib-bibtex",
]
EXAMPLES = [
    "matplotlib",
    "scikit-learn",
]
TESTS = [
    "codecov",
    "pytest",
]
ALL = CLI + DEV + DOCS + EXAMPLES + TESTS

setup(
    name=safe_name("prototorch_" + PLUGIN_NAME),
    version="0.4.0",
    description="Pre-packaged prototype-based "
    "machine learning models using ProtoTorch and PyTorch-Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Engelsberger",
    author_email="engelsbe@hs-mittweida.de",
    url=PROJECT_URL,
    download_url=DOWNLOAD_URL,
    license="MIT",
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV,
        "examples": EXAMPLES,
        "tests": TESTS,
        "all": ALL,
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "prototorch.plugins": f"{PLUGIN_NAME} = prototorch.{PLUGIN_NAME}"
    },
    packages=find_namespace_packages(include=["prototorch.*"]),
    zip_safe=False,
)
