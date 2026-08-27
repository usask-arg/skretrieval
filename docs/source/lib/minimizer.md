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

`SciPyMinimizer` can be configured with `jacobian_mode="matrix_free"` to use SASKTRAN2 linearization products
instead of materializing the full measurement Jacobian. This requires a SASKTRAN2 version that provides
`Engine.linearize()`, `jvp()`, and `vjp()`; unsupported versions and custom transforms produce a clear error in
strict mode. `jacobian_mode="auto"` attempts the product path and warns before falling back to the legacy
materialized path.

The convenience option `minimizer="scipy_lsmr"` uses a SciPy `LinearOperator` and the LSMR trust-region solver.
Pass controls such as `tr_options={"atol": 1e-4, "btol": 1e-4}` or `tr_options={"maxiter": 10}` through
`minimizer_kwargs` to limit inner JVP/VJP calls. It skips diagnostics by default. Full matrix-free diagnostics form
the posterior information matrix with repeated products and may still be expensive; request them explicitly with
`matrix_free_diagnostics="full"` when needed.

The convenience option `minimizer="scipy_lbfgsb"` uses gradient-only L-BFGS-B, requiring VJP products but no JVP
products during the solve. It defaults to 100 function evaluations, `ftol=1e-8`, no matrix-free diagnostics, and
`target_kwargs={"rescale_state_space": True}`. The bounded internal parameterization prevents line-search trials
directly at extreme physical bounds. Each default can be overridden through `minimizer_kwargs` or `target_kwargs`.

For benchmarking or validation against SASKTRAN2's lazy linearization materializer, keep
`minimizer="scipy"` and pass `materialized_jacobian_source="linearization"` in `minimizer_kwargs`.
The default, `materialized_jacobian_source="calculate_radiance"`, preserves the legacy dense
weighting-function path.


## Available Minimizers
```{eval-rst}
.. autosummary::
    skretrieval.retrieval.rodgers.Rodgers
    skretrieval.retrieval.scipy.SciPyMinimizer
```
