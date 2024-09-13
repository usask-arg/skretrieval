(_forward_model)=
# Forward Models
A forward model in `skretrieval` is something that simulates the observation data, given the state vector.
This is accomplished through the {py:class}`skretrieval.retrieval.ForwardModel` class
and the `calculate_radiance()` method.


## Available Forward Models
`skretrieval` includes some pre-built forward models that are suitable for general instruments.  We recommend
determining if your problem is suited to one of these pre-built objects before creating your
own forward model.

```{eval-rst}
.. autosummary::
    skretrieval.forwardmodel.IdealViewingSpectrograph
```

## The Default Forward Model
If no options are specified when construcing the `Retrieval` object,
the default forward model, {py:class}`skretrieval.forwardmodel.IdealViewingSpectrograph` object, is created.
This behaviour can be explicitly set by passing

```python
forward_model_kwargs = {"class": skretrieval.forwardmodel.IdealViewingSpectrograph}
```

to the retrieval.

The ideal viewing component indicates that no integration is performed over the viewing dimension, i.e.,
the spatial point spread function is assumed to be a delta function.
For the spectral dimension, the default is also to assume delta function behaviour, but this can be changed
by specifying extra parameters, for example to switch to a Gaussian line shape,

```python
from skretrieval.retrieval.core.lineshape import Gaussian

forward_model_kwargs = {"class": skretrieval.forwardmodel.IdealViewingSpectrograph,
                        "line_shape_fn": lambda w: Gaussian(fwhm=1.5),
                        "model_res_nm": 0.1
                        }
```

Since the lineshape may be a function of wavelength, we have to specify a function that takes in
wavelength and returns back the lineshape rather than just a singular lineshape.
In addition we pass in the `model_res_nm` parameter which indicates the resolution the forward model
should simulate the radiance at.

## Configuring the Radiative Transfer Model
Every forward model is passed a {py:class}`sasktran2.Config` object that may be used to configure
various radiative transfer settings.  These settings are configured through the `model_kwargs`
parameter that is passed to the {py:class}`skretrieval.Retrieval` object. For example,

```python
import sasktran2 as sk
import skretrieval as skr

model_kwargs = {
    "num_threads": 8,
    "multiple_scatter_source": sk.MultipleScatterSource.DiscreteOrdinates
}

ret = skr.Retrieval(..., model_kwargs=model_kwargs)
```

This is equivalent to creating a {py:class}`sasktran2.Config` object with the properties

```python
config = sk.Config()

config.num_threads = 8
config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
```

Setting the radiative transfer settings is something that is almost almost problem
dependent, and therefore the defaults that `skretrieval` uses are likely not
suitable for your specific problem.  Therefore we always recommend
