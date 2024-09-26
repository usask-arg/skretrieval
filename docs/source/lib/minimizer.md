---
file_format: mystnb
---

(_minimizer)=
# Minimizers
In `skretrieval` a minimizer is the object responsible for determining the optimal
state vector that minimizes the differences between the real measurements and the simulated measurements.
The default minimizer is the {py:class}`skretrieval.retrieval.rodgers.Rodgers` object that is an in-house implementation
of the inversion methods described in "Inverse Methods for Atmospheric Sounding" by Clive Rodgers.  Iteration is performed
using a Levenberg-Marquardt technique.  This is generally a good choice for most problems, and has quite a few diagnostics built
into the calculation.  Using an in-house technique also allows for inspection of the retrieval in-between iterations which
can be useful for debugging purposes.

The other minimizer that is available is the {py:class}`skretrieval.retrieval.scipy.SciPyMinimizer` which is a wrapper
around the {py:func}`scipy.optimize.least_squares` function.  You should consider using this minimizer instead
if your problem is highly non-linear, or has many state vector elements that should remain bounded.


## Available Minimizers
```{eval-rst}
.. autosummary::
    skretrieval.retrieval.rodgers.Rodgers
    skretrieval.retrieval.scipy.SciPyMinimizer
```
