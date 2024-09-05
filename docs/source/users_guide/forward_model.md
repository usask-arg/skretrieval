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
