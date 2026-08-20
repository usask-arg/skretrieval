from __future__ import annotations


def _altitude_for_state(results, state: str):
    data = results[state]
    for dim in data.dims:
        if "altitude" in dim:
            return results[dim]

    msg = f"Could not determine altitude coordinate for state '{state}'"
    raise ValueError(msg)


def plot_state(results: dict, state: str, **kwargs):
    import matplotlib.pyplot as plt

    r = results["state"]
    altitude = _altitude_for_state(r, state)

    plt.subplot(1, 2, 1)
    plt.plot(r[f"{state}"], altitude)
    plt.plot(r[f"{state}_prior"], altitude)
    plt.legend(["Retrieved", "Prior"])

    plt.ylabel("Altitude [m]")
    plt.xlabel(f"{state}")

    plt.subplot(1, 2, 2)
    plt.plot(r[f"{state}_averaging_kernel"], altitude)
    plt.xlabel("Averaging Kernel")
    plt.ylabel("Altitude [m]")

    plt.tight_layout()

    if kwargs.get("show", False):
        plt.show()
