from __future__ import annotations


def plot_state(results: dict, state: str, **kwargs):
    import matplotlib.pyplot as plt

    r = results["state"]
    plt.subplot(1, 2, 1)
    plt.plot(r[f"{state}"], r["altitude"])
    plt.plot(r[f"{state}_prior"], r["altitude"])
    plt.legend(["Retrieved", "Prior"])

    plt.ylabel("Altitude [m]")
    plt.xlabel(f"{state}")

    plt.subplot(1, 2, 2)
    plt.plot(r[f"{state}_averaging_kernel"], r["altitude"])
    plt.xlabel("Averaging Kernel")
    plt.ylabel("Altitude [m]")

    plt.tight_layout()

    if kwargs.get("show", False):
        plt.show()
