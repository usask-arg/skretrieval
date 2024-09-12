(_forward_model)=
# Forward Models
A forward model in `usarm` is something that simulates the observation data, given the state vector.
This is accomplished through the {py:class}`skretrieval.usarm.forwardmodel.USARMForwardModel` class
and the `calculate_radiance()` method.


## Available Forward Models
`usarm` includes some pre-built forward models that are suitable for general instruments.  We recommend
determining if your problem is suited to one of these pre-built objects before creating your
own forward model.

```{eval-rst}
.. autosummary::
    skretrieval.usarm.forwardmodel.IdealViewingSpectrograph
```

## The Default Forward Model
If no options are specified when construcing the `USARMRetrieval` object,
the default forward model, {py:class}`skretrieval.usarm.forwardmodel.IdealViewingSpectrograph` object, is created.
This behaviour can be explicitly set by passing

```python
forward_model_kwargs = {"class": skretrieval.usarm.forwardmodel.IdealViewingSpectrograph}
```

to the retrieval.

The ideal viewing component indicates that no integration is performed over the viewing dimension, i.e.,
the spatial point spread function is assumed to be a delta function.
For the spectral dimension, the default is also to assume delta function behaviour, but this can be changed
by specifying extra parameters, for example to switch to a Gaussian line shape,

```python
from skretrieval.retrieval.core.lineshape import Gaussian

forward_model_kwargs = {"class": skretrieval.usarm.forwardmodel.IdealViewingSpectrograph,
                        "line_shape_fn": lambda w: Gaussian(fwhm=1.5),
                        "model_res_nm": 0.1
                        }
```

Since the lineshape may be a function of wavelength, we have to specify a function that takes in
wavelength and returns back the lineshape rather than just a singular lineshape.
In addition we pass in the `model_res_nm` parameter which indicates the resolution the forward model
should simulate the radiance at.
