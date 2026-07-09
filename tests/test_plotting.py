from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from skretrieval.plotting import plot_state


def test_plot_state_uses_state_specific_altitude_coordinate():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    retrieval_grid = np.array([0.0, 10000.0, 20000.0])
    model_grid = np.array([0.0, 5000.0, 10000.0, 15000.0, 20000.0])
    state = xr.Dataset(
        {
            "o3_vmr": xr.DataArray(
                np.array([1.0, 2.0, 3.0]),
                dims=["o3_altitude"],
                coords={"o3_altitude": retrieval_grid},
            ),
            "o3_vmr_prior": xr.DataArray(
                np.array([1.5, 2.5, 3.5]),
                dims=["o3_altitude"],
                coords={"o3_altitude": retrieval_grid},
            ),
            "o3_vmr_averaging_kernel": xr.DataArray(
                np.eye(len(retrieval_grid)),
                dims=["o3_altitude", "o3_altitude_2"],
                coords={
                    "o3_altitude": retrieval_grid,
                    "o3_altitude_2": retrieval_grid,
                },
            ),
        },
        coords={"altitude": model_grid},
    )

    plot_state({"state": state}, "o3_vmr")

    axes = plt.gcf().axes
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), retrieval_grid)
    np.testing.assert_allclose(axes[0].lines[1].get_ydata(), retrieval_grid)
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), retrieval_grid)
    plt.close("all")
