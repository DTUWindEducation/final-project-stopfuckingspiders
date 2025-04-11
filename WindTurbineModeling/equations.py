import numpy as np
import pandas as pd

def calc_flow_angle(bc):
    """
    Calculate the flow angle phi [rad]
    phi = arctan((1 - a) * V_0 / ((1 + a') * omega * r))

    Parameters:
        bc (BoundaryConditions): Boundary conditions

    Returns:
        phi (pd.Series or float): Flow angle in radians
    """
    a = bc.a0
    a_prime = bc.a0_prime
    V_0 = bc.V_0
    omega = bc.omega
    r = bc.r
    phi = np.arctan((1 - a) * V_0 / ((1 + a_prime) * omega * r))
    return phi

def calc_local_angle_of_attack(phi_rad, theta_p_deg, beta_deg):
    """
    Calculate local angle of attack in degrees.

    alpha = phi [deg] - (pitch + twist)

    Parameters:
        phi_rad (float or pd.Series): Flow angle in radians
        theta_p_deg (pd.Series): Blade pitch angle in degrees
        beta_deg (float): Local twist angle in degrees

    Returns:
        alpha_deg (pd.Series): Local angle of attack in degrees
    """
    alpha_deg = np.rad2deg(phi_rad) - (theta_p_deg + beta_deg)
    return alpha_deg

def calc_local_lift_drag_force(alpha_deg, df):
    """
    Interpolate lift and drag coefficients from airfoil data, with clamping.

    Parameters:
        alpha_deg (pd.Series): Local angle of attack in degrees
        df (pd.DataFrame): Airfoil data with 'Alpha (deg)', 'Cl', 'Cd'

    Returns:
        loc_C_d (pd.Series): Drag coefficient
        loc_C_l (pd.Series): Lift coefficient
    """
    alpha_clamped = np.clip(alpha_deg, df['Alpha (deg)'].min(), df['Alpha (deg)'].max())
    loc_C_d = pd.Series(np.interp(alpha_clamped, df['Alpha (deg)'], df['Cd']), name="C_d [-]")
    loc_C_l = pd.Series(np.interp(alpha_clamped, df['Alpha (deg)'], df['Cl']), name="C_l [-]")
    return loc_C_d, loc_C_l

def calc_local_solidity(bc, loc_BlChord):
    """
    Compute local solidity.

    Parameters:
        bc (BoundaryConditions)
        loc_BlChord (float): Local chord length [m]

    Returns:
        sigma (float): Local solidity [-]
    """
    return (loc_BlChord * bc.Num_Blades) / (2 * np.pi * bc.r)

def calc_normal_tangential_constants(phi_rad, C_d, C_l):
    """
    Compute normal and tangential aerodynamic force coefficients.

    Parameters:
        phi_rad (float or pd.Series): Flow angle in radians
        C_d (pd.Series): Drag coefficient
        C_l (pd.Series): Lift coefficient

    Returns:
        C_n (pd.Series): Normal force coefficient
        C_t (pd.Series): Tangential force coefficient
    """
    C_n = C_l * np.cos(phi_rad) + C_d * np.sin(phi_rad)
    C_t = C_l * np.sin(phi_rad) + C_d * np.cos(phi_rad)
    return C_n, C_t

def update_induction_factors(phi_rad, sigma, C_n, C_t):
    """
    Compute updated axial and tangential induction factors.
    Clamps denominator to avoid division by zero or instability.

    Parameters:
        phi_rad (pd.Series): Flow angle in radians
        sigma (float): Local solidity
        C_n (pd.Series): Normal force coefficient
        C_t (pd.Series): Tangential force coefficient

    Returns:
        a (pd.Series): Axial induction factor
        a_prime (pd.Series): Tangential induction factor
    """
    eps = 1e-6

    # Axial
    denom_a = (4 * np.sin(phi_rad)**2) / (sigma * C_n + 1)
    denom_a = np.where(denom_a == 0, eps, denom_a)
    a = 1 / denom_a

    # Tangential
    denom_aprime = (4 * np.sin(phi_rad) * np.cos(phi_rad)) / np.clip((sigma * C_t - 1), eps, None)
    a_prime = 1 / denom_aprime

    return a, a_prime

def compute_local_thrust(r_series, V_0, a, RHO, dr):
    """
    Compute local thrust dT.

    Returns:
        dT (pd.Series): Local thrust [N]
    """
    return 4 * np.pi * r_series * RHO * V_0**2 * a * (1 - a) * dr

def compute_local_torque(r_series, V_0, a, a_prime, omega, RHO, dr):
    """
    Compute local torque dM.

    Returns:
        dM (pd.Series): Local torque [Nm]
    """
    return 4 * np.pi * r_series**3 * RHO * V_0 * omega * a_prime * (1 - a) * dr

def interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord):
    """
    Interpolate twist and chord at radius r.

    Returns:
        twist (float): Interpolated twist angle [deg]
        chord (float): Interpolated chord length [m]
    """
    twist = np.interp(r, BlSpn, BlTwist)
    chord = np.interp(r, BlSpn, BlChord)
    return twist, chord

def compute_rotor_coefficients(T, P, RHO, R, V_0):
    """
    Compute thrust and power coefficients.

    Returns:
        C_T, C_P (floats): Non-dimensional coefficients
    """
    A = np.pi * R**2
    V0_mean = V_0.mean()
    C_T = T / (0.5 * RHO * A * V0_mean**2)
    C_P = P / (0.5 * RHO * A * V0_mean**3)
    return C_T, C_P

def compute_totals_and_coefficients(dT, dM, omega, dr, RHO, R, V_0):
    """
    Compute total thrust, torque, power, and coefficients using trapezoidal integration.

    Returns:
        T, M, P, C_T, C_P (floats)
    """
    T = np.trapz(dT, dx=dr)
    M = np.trapz(dM, dx=dr)
    P = np.trapz(omega * dM / dr, dx=dr)
    C_T, C_P = compute_rotor_coefficients(T, P, RHO, R, V_0)
    return T, M, P, C_T, C_P

def print_summary(T, M, P, C_T, C_P):
    """
    Print final rotor performance summary.
    """
    print(f"Thrust (T): {T:.2f} N")
    print(f"Torque (M): {M:.2f} Nm")
    print(f"Power (P): {P:.2f} W")
    print(f"Thrust Coefficient (C_T): {C_T:.4f}")
    print(f"Power Coefficient (C_P): {C_P:.4f}")
