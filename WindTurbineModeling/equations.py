import numpy as np
import pandas as pd

def calc_flow_angle(a, a_prime, V_0, omega, r):
    """
    phi [deg] or [rad]:      Flow angle
    a [?]:          Axial induction factor
    a_prime [?]:    Tangential induction factor
    V_0 [m/s]:      Inflow wind speed
    omega [1/s]:    Rotational speed
    r [m]:          Span position
    """
    #TODO finish DOGSTRING
    # Calculate the flow angle using the given formula
    # phi = arctan((1-a)*V_0 / ((1+a')*omega*r))
    numerator = (1 - a) * V_0
    denominator = (1 + a_prime) * omega * r
    phi = np.arctan(numerator / denominator)

    return phi

def calc_local_angle_of_attack(phi, theta_p, beta):
    """
    alpha [deg]: local angle of attack
    phi [deg]: Flow angle
    theta_p [deg]: Blade_pitch_angle
    beta [deg]: local twist angle
    r [m]: Span position
    """
    #TODO finish DOGSTRING

    alpha = phi  - (theta_p + beta)
    return alpha

def calc_local_lift_drag_force(alpha, df):
    # Get twist angle form aerodynamic coefficients
    #TODO add DOGSTRING
    C_d = df['Cd']
    C_l = df['Cl']
   
    res = np.interp(alpha,  df['Alpha (deg)'], C_d) #[deg]
    loc_C_d = pd.Series(res, name="C_l [-]")

    res = np.interp(alpha,  df['Alpha (deg)'], C_l) #[deg]
    loc_C_l = pd.Series(res, name="C_d [-]")

    return loc_C_d, loc_C_l

def calc_local_solidity(r, c, B):
    #TODO add DOGSTRING
    sigma = (c * B)/(2 * np.pi * r)
    return sigma
