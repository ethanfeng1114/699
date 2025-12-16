import pybamm
import pandas as pd
import numpy as np
from pybamm.input.parameters.lithium_ion.Xu2019 import (
    electrolyte_diffusivity_Valoen2005,
    electrolyte_conductivity_Valoen2005,
)


a_str = """-0.05603759 -0.07708655 -0.11896403 -0.09649386  0.51492991 -0.09382906
 -1.48642653  0.53330793  1.90199907 -0.60285044 -1.02723306 -0.1533813 """

c_str = """-0.28623998 -0.04243566  0.1091517  -0.01585551 -0.57695251  0.05901815
  1.7448597   0.12388956 -2.12823709 -0.17531718  0.93339131 -3.88355819"""




anode_j_df = pd.read_csv("P45B/anode_exchange_current.txt", header=None, sep=" ")
anode_j_df.dropna(axis=1, inplace=True)
cathode_j_df = pd.read_csv(
    "P45B/cathode_exchange_current.txt", header=None, sep="  "
)
cathode_j_df.dropna(axis=1, inplace=True)


def p45b_anode_ocv(sto):
    if sto == 0:
        sto += 1e-5
    elif sto == 1:
        sto -= 1e-5

    coeffs = np.array([float(x) for x in a_str.split()])

    result = 0.0
    for i in range(0, len(coeffs) - 1):
        result += (coeffs[i]) * (
            (1 - 2 * sto) ** (i + 1)
            - (2 * sto * i * (1 - sto)) / (1 - 2 * sto) ** (1 - i)
        )

    result += (8.3145 * 298.15 / 96485.3321) * np.log(sto / (1 - sto))
    result += coeffs[-1]

    return -result


def p45b_cathode_ocv(sto):
    if sto == 0:
        sto += 1e-5
    elif sto == 1:
        sto -= 1e-5

    coeffs = np.array([float(x) for x in c_str.split()])

    result = 0.0
    for i in range(0, len(coeffs) - 1):
        result += (coeffs[i]) * (
            (1 - 2 * sto) ** (i + 1)
            - (2 * sto * i * (1 - sto)) / (1 - 2 * sto) ** (1 - i)
        )

    result += (8.3145 * 298.15 / 96485.3321) * np.log(sto / (1 - sto))
    result += coeffs[-1]

    return -result


def anode_entropic_change(sto):
    anode_dOCVdT_df = pd.read_csv(
        "P45B/anode_entropic_coefficient.txt", header=None, sep=" "
    )
    anode_dOCVdT_df.dropna(axis=1, inplace=True)
    data = anode_dOCVdT_df
    return pybamm.Interpolant(
        data[data.columns[0]].values, data[data.columns[1]].values, sto
    )


def cathode_entropic_change(sto):
    cathode_dOCVdT_df = pd.read_csv(
        "P45B/cathode_entropic_coefficient.txt", header=None, sep="  "
    )
    cathode_dOCVdT_df.dropna(axis=1, inplace=True)
    data = cathode_dOCVdT_df
    return pybamm.Interpolant(
        data[data.columns[0]].values[::-1], data[data.columns[1]].values[::-1], sto
    )


def graphite_diffusivity_Kim2011(sto, T):
    """
    Graphite diffusivity [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    # TODO: Silicon graphite not graphite

    D_ref = 9 * 10 ** (-14)
    E_D_s = 4e3
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_ocp_Kim2011(sto):
    """
    Graphite Open-circuit Potential (OCP) as a function of the stoichiometry [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.
    """

    u_eq = (
        0.124
        + 1.5 * np.exp(-70 * sto)
        - 0.0351 * np.tanh((sto - 0.286) / 0.083)
        - 0.0045 * np.tanh((sto - 0.9) / 0.119)
        - 0.035 * np.tanh((sto - 0.99) / 0.05)
        - 0.0147 * np.tanh((sto - 0.5) / 0.034)
        - 0.102 * np.tanh((sto - 0.194) / 0.142)
        - 0.022 * np.tanh((sto - 0.98) / 0.0164)
        - 0.011 * np.tanh((sto - 0.124) / 0.0226)
        + 0.0155 * np.tanh((sto - 0.105) / 0.029)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Kim2011(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC
    [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    i0_ref = 36  # reference exchange current density at 100% SOC
    sto = 0.36  # stoichiometry at 100% SOC
    c_s_n_ref = sto * c_s_max  # reference electrode concentration
    c_e_ref = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")
    alpha = 0.5  # charge transfer coefficient

    m_ref = i0_ref / (
        c_e_ref**alpha * (c_s_max - c_s_n_ref) ** alpha * c_s_n_ref**alpha
    )

    E_r = 3e4
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**alpha * c_s_surf**alpha * (c_s_max - c_s_surf) ** alpha
    )


def nca_diffusivity_Kim2011(sto, T):
    """
    NCA diffusivity as a function of stoichiometry [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    D_ref = 3 * 10 ** (-15)
    E_D_s = 2e4
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def nca_electrolyte_exchange_current_density_Kim2011(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NCA and LiPF6 in EC:DMC
    [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    i0_ref = 4  # reference exchange current density at 100% SOC
    sto = 0.41  # stoichiometry at 100% SOC
    c_s_ref = sto * c_s_max  # reference electrode concentration
    c_e_ref = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")
    alpha = 0.5  # charge transfer coefficient

    m_ref = i0_ref / (c_e_ref**alpha * (c_s_max - c_s_ref) ** alpha * c_s_ref**alpha)
    E_r = 3e4
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**alpha * c_s_surf**alpha * (c_s_max - c_s_surf) ** alpha
    )


def electrolyte_diffusivity_Kim2011(c_e, T):
    """
    Diffusivity of LiPF6 in EC as a function of ion concentration from [1].

     References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = (
        5.84 * 10 ** (-7) * np.exp(-2870 / T) * (c_e / 1000) ** 2
        - 33.9 * 10 ** (-7) * np.exp(-2920 / T) * (c_e / 1000)
        + 129 * 10 ** (-7) * np.exp(-3200 / T)
    )

    return D_c_e


def electrolyte_conductivity_Kim2011(c_e, T):
    """
    Conductivity of LiPF6 in EC as a function of ion concentration from [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        3.45 * np.exp(-798 / T) * (c_e / 1000) ** 3
        - 48.5 * np.exp(-1080 / T) * (c_e / 1000) ** 2
        + 244 * np.exp(-1440 / T) * (c_e / 1000)
    )

    return sigma_e


def nca_ocp_Kim2011(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the stoichiometry [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.
    """

    U_posi = (
        1.638 * sto**10
        - 2.222 * sto**9
        + 15.056 * sto**8
        - 23.488 * sto**7
        + 81.246 * sto**6
        - 344.566 * sto**5
        + 621.3475 * sto**4
        - 554.774 * sto**3
        + 264.427 * sto**2
        - 66.3691 * sto
        + 11.8058
        - 0.61386 * np.exp(5.8201 * sto**136.4)
    )

    return U_posi


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a "Nominal Design" graphite/NCA pouch cell, from the paper
    :footcite:t:`Kim2011`

    .. note::
        Only an effective cell volumetric heat capacity is provided in the paper. We
        therefore used the values for the density and specific heat capacity reported in
        the Marquis2019 parameter set in each region and multiplied each density by the
        ratio of the volumetric heat capacity provided in smith to the calculated value.
        This ensures that the values produce the same effective cell volumetric heat
        capacity. This works fine for thermal models that are averaged over the
        x-direction but not for full (PDE in x direction) thermal models. We do the same
        for the planar effective thermal conductivity.

    SEI parameters are example parameters for SEI growth from the papers
    :footcite:t:`Ramadass2004`, :footcite:t:`Ploehn2004`,
    :footcite:t:`Single2018`, :footcite:t:`Safari2008`, and
    :footcite:t:`Yang2017`

    .. note::
        This parameter set does not claim to be representative of the true parameter
        values. Instead these are parameter values that were used to fit SEI models to
        observed experimental data in the referenced papers.
    """

    return pybamm.ParameterValues(
        {
            "chemistry": "lithium_ion",
            # cell
            "Negative current collector thickness [m]": 1e-05,
            "Separator thickness [m]": 1.4e-05,
            "Positive electrode thickness [m]": 4.2e-05,
            "Positive current collector thickness [m]": 1.4e-05,
            "Electrode height [m]": 0.064,
            "Electrode width [m]": 1.32 * 1.77,  # ~ Cell geometry correction
            # "Cell cooling surface area [m2]": 0.004746,
            "Cell volume [m3]": 2.31e-5,
            "Negative current collector conductivity [S.m-1]": 59500000.0,
            "Positive current collector conductivity [S.m-1]": 37700000.0,
            "Negative current collector density [kg.m-3]": 8960,  # Cu
            "Positive current collector density [kg.m-3]": 2700,  # Al
            "Negative current collector specific heat capacity [J.kg-1.K-1]": 390.0,
            "Positive current collector specific heat capacity [J.kg-1.K-1]": 903.0,
            "Negative current collector thermal conductivity [W.m-1.K-1]": 398,
            "Positive current collector thermal conductivity [W.m-1.K-1]": 237,
            "Nominal cell capacity [A.h]": 4.5,
            # "Current function [A]": 0.43,
            "Contact resistance [Ohm]": 0,
            # negative electrode
            "Negative electrode thickness [m]": 5e-05,
            "Negative electrode conductivity [S.m-1]": 38.7,
            "Maximum concentration in negative electrode [mol.m-3]": 29871.0,
            "Negative electrode diffusivity [m2.s-1]": 1.5e-13,
            "Negative electrode OCP [V]": p45b_anode_ocv,
            "Negative electrode porosity": 0.196,
            "Negative electrode active material volume fraction": 0.754,
            "Negative particle radius [m]": 5.11e-06,
            "Negative electrode Bruggeman coefficient (electrolyte)": 2.0,
            "Negative electrode Bruggeman coefficient (electrode)": 2.0,
            "Negative electrode charge transfer coefficient": 0.5,
            "Negative electrode double-layer capacity [F.m-2]": 0.2,
            "Negative electrode exchange-current density [A.m-2]"
            "": graphite_electrolyte_exchange_current_density_Kim2011,
            "Negative electrode density [kg.m-3]": 2136.43638,
            "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
            "Negative electrode thermal conductivity [W.m-1.K-1]": 1.1339,
            "Negative electrode OCP entropic change [V.K-1]": anode_entropic_change,
            # positive electrode
            "Positive electrode conductivity [S.m-1]": 17.3,
            "Maximum concentration in positive electrode [mol.m-3]": 46456.0,
            "Positive electrode diffusivity [m2.s-1]": 1.6e-13,
            "Positive electrode OCP [V]": p45b_cathode_ocv,
            "Positive electrode porosity": 0.227,
            "Positive electrode active material volume fraction": 0.723,
            "Positive particle radius [m]": 7.54e-06,
            "Positive electrode Bruggeman coefficient (electrolyte)": 2.0,
            "Positive electrode Bruggeman coefficient (electrode)": 2.0,
            "Positive electrode charge transfer coefficient": 0.5,
            "Positive electrode double-layer capacity [F.m-2]": 0.2,
            "Positive electrode exchange-current density [A.m-2]"
            "": nca_electrolyte_exchange_current_density_Kim2011,
            "Positive electrode OCP entropic change [V.K-1]": cathode_entropic_change,
            "Positive electrode density [kg.m-3]": 4205.82708,
            "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
            "Positive electrode thermal conductivity [W.m-1.K-1]": 1.4007,
            # separator
            "Separator porosity": 0.4,
            "Separator Bruggeman coefficient (electrolyte)": 2.0,
            "Separator density [kg.m-3]": 511.86798,
            "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
            "Separator thermal conductivity [W.m-1.K-1]": 0.10672,
            # electrolyte
            "Initial concentration in electrolyte [mol.m-3]": 1000.0,
            "Cation transference number": 0.38,
            "Thermodynamic factor": 1.0,
            "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Valoen2005,
            "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Valoen2005,
            # experiment
            "Reference temperature [K]": 298.15,
            "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
            "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
            "Edge heat transfer coefficient [W.m-2.K-1]": 0.7,
            "Total heat transfer coefficient [W.m-2.K-1]": 30.0,
            "Ambient temperature [K]": 298.15,
            "Number of electrodes connected in parallel to make a cell": 1.0,
            "Number of cells connected in series to make a battery": 1.0,
            "Lower voltage cut-off [V]": 2.3,
            "Upper voltage cut-off [V]": 4.5,
            "Open-circuit voltage at 0% SOC [V]": 2.6,
            "Open-circuit voltage at 100% SOC [V]": 4.3,
            "Initial concentration in negative electrode [mol.m-3]": 18081.0,
            "Initial concentration in positive electrode [mol.m-3]": 20090.0,
            "Initial temperature [K]": 298.15,
            "Current function [A]": 0.1,
        }
    )
