import numpy as np

import pybamm
import pandas as pd
from scipy.optimize import minimize


a_str = """-0.13451994 -0.15161015 -0.16027467 -0.36175146  0.14457436  0.63714431
 -0.44595161 -1.04608012 -0.02299912  0.20079712  0.0592579  -0.21038388"""

c_str = """ -6.64535356  -6.18619563  -6.03372075  -6.37885913  -5.24537211
  -3.24156772 -13.14835885 -24.66453109   0.95565813  32.49103059
  19.01226102  -7.06196757"""

def lgm50_anode_ocv(sto):
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


def lgm50_cathode_ocv(sto):
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


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
    m_ref = 3.42e-6  # (A/m2)(m3/mol)**1.5 - includes ref concentrations\
    E_r = 17800
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_diffusivity_Nyman2008(c_e, T):
    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    return pybamm.ParameterValues(
        {
            "chemistry": "lithium_ion",
            # cell
            "Negative current collector thickness [m]": 1.2e-05,
            "Negative electrode thickness [m]": 8.52e-05,
            "Separator thickness [m]": 1.2e-05,
            "Positive electrode thickness [m]": 7.56e-05,
            "Positive current collector thickness [m]": 1.6e-05,
            "Electrode height [m]": 0.065,
            "Electrode width [m]": 1.58,
            "Cell cooling surface area [m2]": 0.00531,
            "Cell volume [m3]": 2.42e-05,
            "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
            "Negative current collector conductivity [S.m-1]": 58411000.0,
            "Positive current collector conductivity [S.m-1]": 36914000.0,
            "Negative current collector density [kg.m-3]": 8960.0,
            "Positive current collector density [kg.m-3]": 2700.0,
            "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
            "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
            "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
            "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
            "Nominal cell capacity [A.h]": 5.0,
            "Current function [A]": 5.0,
            "Contact resistance [Ohm]": 0,
            # negative electrode
            "Negative electrode conductivity [S.m-1]": 215.0,
            "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
            "Negative electrode diffusivity [m2.s-1]": 3.3e-14,
            "Negative electrode porosity": 0.25,
            "Negative electrode OCP [V]": lgm50_anode_ocv,
            "Negative electrode active material volume fraction": 0.75,
            "Negative particle radius [m]": 5.86e-06,
            "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
            "Negative electrode Bruggeman coefficient (electrode)": 0,
            "Negative electrode charge transfer coefficient": 0.5,
            "Negative electrode double-layer capacity [F.m-2]": 0.2,
            "Negative electrode exchange-current density [A.m-2]"
            "": graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
            "Negative electrode density [kg.m-3]": 1657.0,
            "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
            "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
            "Negative electrode OCP entropic change [V.K-1]": 0.0,
            # positive electrode
            "Positive electrode conductivity [S.m-1]": 0.18,
            "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
            "Positive electrode diffusivity [m2.s-1]": 4e-15,
            "Positive electrode porosity": 0.335,
            "Positive electrode active material volume fraction": 0.665,
            "Positive particle radius [m]": 5.22e-06,
            "Positive electrode OCP [V]": lgm50_cathode_ocv,
            "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
            "Positive electrode Bruggeman coefficient (electrode)": 0,
            "Positive electrode charge transfer coefficient": 0.5,
            "Positive electrode double-layer capacity [F.m-2]": 0.2,
            "Positive electrode exchange-current density [A.m-2]"
            "": nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
            "Positive electrode density [kg.m-3]": 3262.0,
            "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
            "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
            "Positive electrode OCP entropic change [V.K-1]": 0.0,
            # separator
            "Separator porosity": 0.47,
            "Separator Bruggeman coefficient (electrolyte)": 1.5,
            "Separator density [kg.m-3]": 397.0,
            "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
            "Separator thermal conductivity [W.m-1.K-1]": 0.16,
            # electrolyte
            "Initial concentration in electrolyte [mol.m-3]": 1000.0,
            "Cation transference number": 0.2594,
            "Thermodynamic factor": 1.0,
            "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008,
            "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Nyman2008,
            # experiment
            "Reference temperature [K]": 298.15,
            "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
            "Ambient temperature [K]": 298.15,
            "Number of electrodes connected in parallel to make a cell": 1.0,
            "Number of cells connected in series to make a battery": 1.0,
            "Lower voltage cut-off [V]": 2.5,
            "Upper voltage cut-off [V]": 4.2,
            "Open-circuit voltage at 0% SOC [V]": 2.5,
            "Open-circuit voltage at 100% SOC [V]": 4.2,
            "Initial concentration in negative electrode [mol.m-3]": 29866.0,
            "Initial concentration in positive electrode [mol.m-3]": 17038.0,
            "Initial temperature [K]": 298.15,
        }
    )
