import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt


def redlich_kister_poly(x, coeffs):
    result = 0.0
    for i in range(0, len(coeffs) - 1):
        result += (coeffs[i]) * (
            (1 - 2 * x) ** (i + 1) - (2 * x * i * (1 - x)) / (1 - 2 * x) ** (1 - i)
        )  # Higher-order terms

    result += (8.3145 * 298.15 / 96485.3321) * np.log(x / (1 - x))
    result += coeffs[-1]

    return result


def loss_function(func, coeffs, x, h):
    predictions = func(x, coeffs)
    return np.sum((predictions - h) ** 2)


def get_coeffs(x_data, y_data, N_max):
    initial_guess_rk = np.ones(N_max) * 0.1
    result_rk = minimize(
        lambda c: loss_function(redlich_kister_poly, c, x_data, y_data),
        initial_guess_rk,
        method="BFGS",
    )
    rk_optimized_coeffs = result_rk.x

    return rk_optimized_coeffs


if __name__ == "__main__":
    # Read in ocp file
    lgm50_anode_ocp_df = pd.read_csv("")
    lgm50_sto_anode, lgm50_ocp_anode = (
        lgm50_anode_ocp_df["sto"].to_numpy(),
        lgm50_anode_ocp_df["ocp"].to_numpy(),
    )

    lgm50_cathode_ocp_df = pd.read_csv("")
    lgm50_sto_cathode, lgm50_ocp_cathode = (
        lgm50_cathode_ocp_df["sto"].to_numpy(),
        lgm50_cathode_ocp_df["ocp"].to_numpy(),
    )

    # Get the coefficients for Nth order RK fit
    lgm50_anode_coeffs = get_coeffs(lgm50_sto_anode, -lgm50_ocp_anode, N_max=)

    # Plot the data and fit to verify
    plt.scatter(lgm50_sto_anode, lgm50_ocp_anode)
    plt.plot(
        lgm50_sto_anode,
        -redlich_kister_poly(lgm50_sto_anode, lgm50_anode_coeffs),
        color="r",
        linestyle="--",
    )
    plt.show()

    lgm50_cathode_coeffs = get_coeffs(lgm50_sto_cathode, -lgm50_ocp_cathode, N_max=)
    plt.scatter(lgm50_sto_cathode, lgm50_ocp_cathode)
    plt.plot(
        lgm50_sto_cathode,
        -redlich_kister_poly(lgm50_sto_cathode, lgm50_cathode_coeffs),
        color="r",
        linestyle="--",
    )
    plt.show()

    print(lgm50_anode_coeffs)
    print(lgm50_cathode_coeffs)
