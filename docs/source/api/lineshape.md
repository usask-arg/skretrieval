(_api_lineshape)=
# Lineshapes

The instrument models apply line shapes with {py:func}`skretrieval.core.lineshape.measurement_matrix`,
which uses [rebasis](https://usask-arg.github.io/rebasis/).  Anywhere a line shape is accepted, a
`rebasis.Space` of kernels can be given instead, which gives access to many more line shapes
(Voigt, FTS, super-Gaussian, tabulated, ...).  The classes below are kept for backwards compatibility.

## Measurement Matrix
```{eval-rst}
.. autosummary::
    :toctree: generated/

    skretrieval.core.lineshape.measurement_matrix

```

## Base Class
```{eval-rst}
.. autosummary::
    :toctree: generated/

    skretrieval.core.lineshape.LineShape

```

## Available Lineshapes
```{eval-rst}
.. autosummary::
    :toctree: generated/

    skretrieval.core.lineshape.Gaussian
    skretrieval.core.lineshape.Rectangle
    skretrieval.core.lineshape.DeltaFunction
    skretrieval.core.lineshape.UserLineShape

```
