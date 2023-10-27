# The University of Saskatchewan Retrieval Framework
[![Anaconda-Server Badge](https://anaconda.org/usask-arg/skretrieval/badges/version.svg)](https://anaconda.org/usask-arg/skretrieval)
[![Available on pypi](https://img.shields.io/pypi/v/skretrieval.svg)](https://pypi.python.org/pypi/skretrieval/)
[![Documentation Status](https://readthedocs.org/projects/skretrieval/badge/?version=latest)](https://skretrieval.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/usask-arg/skretrieval/main.svg)](https://results.pre-commit.ci/latest/github/usask-arg/skretrieval/main)

`skretrieval` is an atmospheric retrieval framework developed at the University of Saskatchewan.
It has been successfully used to retrieved atmospheric properties from satellite instruments such as OSIRIS
and OMPS-LP, as well as numerous.  At it's core, `skretrieval` is an optimal estimation framework built around
the [SASKTRAN Radiative Transfer Model](https://github.com/usask-arg/sasktran).  Future versions of the framework
are being developed around [Version 2 of the SASKTRAN Model](https://github.com/usask-arg/sasktran2).

## Installation
The preferred method to install SASKTRAN2 is through conda package

```
conda install -c usask-arg -c conda-forge skretrieval
```

alternatively through `pip`,

```
pip install skretrieval
```
The package is tested on Python versions >= 3.10, on Windows/Linux/MacOS platforms.

## Usage
Documentation can be found at https://sasktran2.readthedocs.io/

## License
skretrieval is made available under the MIT license subject to the Commons Clause condition (see license.md). Effectively this is a MIT license restricted for academic, educational, and/or non-profit use, for commercial use please contact the package authors. Commerical level support may also be available for specific applications.
