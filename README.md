# The University of Saskatchewan Retrieval Framework
[![Available on pypi](https://img.shields.io/pypi/v/skretrieval.svg)](https://pypi.python.org/pypi/skretrieval/)
[![Documentation Status](https://readthedocs.org/projects/skretrieval/badge/?version=latest)](https://skretrieval.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/usask-arg/skretrieval/main.svg)](https://results.pre-commit.ci/latest/github/usask-arg/skretrieval/main)

`skretrieval` is an atmospheric retrieval framework developed at the University of Saskatchewan.
It has been successfully used to retrieve atmospheric properties from satellite instruments such as OSIRIS
and OMPS-LP, as well as numerous other instruments.  At its core, `skretrieval` is an optimal estimation framework built around
the [SASKTRAN2 Radiative Transfer Model](https://github.com/usask-arg/sasktran2).  Support for the original
[SASKTRAN model](https://github.com/usask-arg/sasktran) is available through the `skretrieval.legacy` module.

## Installation
```
pip install skretrieval
```
The package is tested on Python versions >= 3.11, on Windows/Linux/MacOS platforms.

## Usage
Documentation can be found at https://skretrieval.readthedocs.io/

## License
skretrieval is made available under the MIT license.
