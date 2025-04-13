import numpy as np
import pandas as pd

def calc_flow_angle(bc):
    """
    Calculate the local flow angle φ [rad].

    Equation:
        φ = arctan((1 - a) * V₀ / ((1 + a') * ω * r))

    Parameters:
        bc (BoundaryConditions): Boundary conditions including wind speed, rotation, radius, and induction factors.

    Returns:
        float or pd.Series: Flow angle φ in radians.
    """
    a = bc.a0
    a_prime = bc.a0_prime
    V_0 = bc.V_0
    omega = bc.omega
    r = bc.r
    return np.arctan((1 - a) * V_0 / ((1 + a_prime) * omega * r))


def calc_local_angle_of_attack(phi_rad, theta_p_deg, beta_deg):
    """
    Calculate the local angle of attack α [deg].

    Equation:
        α = φ - (θ_p + β)

    Parameters:
        phi_rad (float or pd.Series): Flow angle [rad].
        theta_p_deg (pd.Series): Blade pitch angle [deg].
        beta_deg (float): Local twist angle [deg].

    Returns:
        pd.Series: Local angle of attack α [deg].
    """
    return np.rad2deg(phi_rad) - (theta_p_deg + beta_deg)


def interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord):
    """
    Interpolate local twist and chord length from spanwise distributions.

    Parameters:
        r (float): Radial location [m].
        BlSpn (pd.Series): Blade span positions [m].
        BlTwist (pd.Series): Blade twist angles [deg].
        BlChord (pd.Series): Blade chord lengths [m].

    Returns:
        tuple[float, float]: (twist angle [deg], chord length [m])
    """
    twist = np.interp(r, BlSpn, BlTwist)
    chord = np.interp(r, BlSpn, BlChord)
    return twist, chord


def calc_local_lift_drag_force(alpha, df):
    """
    Interpolate lift and drag coefficients from airfoil polar data.

    Parameters:
        alpha (pd.Series): Angle of attack [deg].
        df (pd.DataFrame): Airfoil data with 'Alpha (deg)', 'Cl', 'Cd'.

    Returns:
        tuple[pd.Series, pd.Series]: (drag coefficient Cd [-], lift coefficient Cl [-])
    """
    alpha_clamped = np.clip(alpha, df['Alpha (deg)'].min(), df['Alpha (deg)'].max())
    Cd = pd.Series(np.interp(alpha_clamped, df['Alpha (deg)'], df['Cd']), name="C_d [-]")
    Cl = pd.Series(np.interp(alpha_clamped, df['Alpha (deg)'], df['Cl']), name="C_l [-]")
    return Cd, Cl


def calc_normal_tangential_constants(phi_rad, C_d, C_l):
    """
    Calculate normal and tangential force coefficients.

    Parameters:
        phi_rad (float or pd.Series): Flow angle [rad].
        C_d (pd.Series): Drag coefficient.
        C_l (pd.Series): Lift coefficient.

    Returns:
        tuple[pd.Series, pd.Series]: (C_n, C_t)
    """
    C_n = C_l * np.cos(phi_rad) + C_d * np.sin(phi_rad)
    C_t = C_l * np.sin(phi_rad) - C_d * np.cos(phi_rad)
    return C_n, C_t


def calc_local_solidity(bc, loc_BlChord):
    """
    Compute local blade solidity σ.

    Parameters:
        bc (BoundaryConditions): Boundary condition context.
        loc_BlChord (float): Local chord length [m].

    Returns:
        float: Local solidity σ [-].
    """
    return (loc_BlChord * bc.Num_Blades) / (2 * np.pi * bc.r)


def calc_prandtl_tip_loss(B, R, r, phi):
    """
    Calculate Prandtl's tip loss factor F.

    Parameters:
        B (int): Number of blades.
        R (float): Rotor radius [m].
        r (float): Local radius [m].
        phi (float): Flow angle [rad].

    Returns:
        float: Tip loss correction factor F [-], clipped between [1e-3, 1.0].
    """
    f = (B / 2) * (R - r) / (r * np.sin(phi))
    f = np.maximum(f, 1e-6)
    F = (2 / np.pi) * np.arccos(np.exp(-f))
    return np.clip(F, 1e-3, 1.0)


def update_induction_factors(phi_rad, sigma, C_n, C_t, F, a_old):
    """
    Update axial (a) and tangential (a') induction factors using Glauert correction and Prandtl's factor.

    Parameters:
        phi_rad (float): Flow angle [rad].
        sigma (float): Local solidity [-].
        C_n (pd.Series): Normal force coefficient.
        C_t (pd.Series): Tangential force coefficient.
        F (float): Prandtl's tip loss factor.
        a_old (float): Previous axial induction factor.

    Returns:
        tuple[float, float]: (updated axial a, tangential a' induction factors)
    """
    eps = 1e-6
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    corr = 0.1  # smoothing factor

    if a_old <= 1/3:
        C_T = 4 * a_old * F * (1 - a_old)
        a_star = (1 - a_old) * (sigma * C_n.mean()) / (4 * F * sin_phi**2 + eps)
    else:
        C_T = ((1 - a_old)**2 * C_n.mean() * sigma) / (sin_phi**2 + eps)
        a_star = C_T / (4 * F * (1 - 0.25 * (5 - 3 * a_old) * a_old) + eps)

    a_new = corr * a_star + (1 - corr) * a_old
    a_prime_star = (1 - a_old) * sigma * C_t.mean() / (4 * F * sin_phi * cos_phi + eps)
    a_prime_new = corr * a_prime_star  # assume a_prime initially 0

    return a_new, a_prime_new


def compute_local_thrust(r_series, V_0, a, RHO, dr):
    """
    Compute differential thrust dT per element.

    Parameters:
        r_series (pd.Series): Local radial position [m].
        V_0 (float): Wind speed [m/s].
        a (float): Axial induction factor.
        RHO (float): Air density [kg/m³].
        dr (float): Radial increment [m].

    Returns:
        pd.Series: Local thrust force [N].
    """
    return 4 * np.pi * r_series * RHO * V_0**2 * a * (1 - a) * dr


def compute_local_torque(r_series, V_0, a, a_prime, omega, RHO, dr):
    """
    Compute differential torque dM per element.

    Parameters:
        r_series (pd.Series): Local radial position [m].
        V_0 (float): Wind speed [m/s].
        a (float): Axial induction factor.
        a_prime (float): Tangential induction factor.
        omega (float): Rotational speed [rad/s].
        RHO (float): Air density [kg/m³].
        dr (float): Radial increment [m].

    Returns:
        pd.Series: Local torque [Nm].
    """
    return 4 * np.pi * r_series**3 * RHO * V_0 * omega * a_prime * (1 - a) * dr


def compute_rotor_coefficients(T, P, RHO, R, V_0):
    """
    Compute non-dimensional thrust and power coefficients.

    Parameters:
        T (float): Total thrust [N].
        P (float): Total power [W].
        RHO (float): Air density [kg/m³].
        R (float): Rotor radius [m].
        V_0 (pd.Series): Wind speed [m/s].

    Returns:
        tuple[float, float]: (C_T, C_P)
    """
    A = np.pi * R**2
    V0_mean = V_0.mean()
    C_T = T / (0.5 * RHO * A * V0_mean**2)
    C_P = P / (0.5 * RHO * A * V0_mean**3)
    return C_T, C_P


def compute_totals_and_coefficients(dT, dM, omega_scalar, dr, RHO, R, V_0):
    """
    Compute total thrust, torque, power, and rotor coefficients.

    Parameters:
        dT (pd.Series): Local thrust values [N].
        dM (pd.Series): Local torque values [Nm].
        omega_scalar (float): Rotational speed [rad/s].
        dr (float): Radial resolution [m].
        RHO (float): Air density [kg/m³].
        R (float): Rotor radius [m].
        V_0 (pd.Series): Inflow wind speed [m/s].

    Returns:
        tuple[float, float, float, float, float]: (T, M, P, C_T, C_P)
    """
    T = np.trapz(dT, dx=dr)
    M = np.trapz(dM, dx=dr)
    P = np.trapz((dM / dr).values * omega_scalar, dx=dr)
    C_T, C_P = compute_rotor_coefficients(T, P, RHO, R, V_0)
    return T, M, P, C_T, C_P


def print_summary(T, M, P, C_T, C_P):
    """
    Print final rotor performance summary.

    Parameters:
        T (float): Thrust [N].
        M (float): Torque [Nm].
        P (float): Power [W].
        C_T (float): Thrust coefficient.
        C_P (float): Power coefficient.
    """
    print(f"Thrust (T): {T:.2f} N")
    print(f"Torque (M): {M:.2f} Nm")
    print(f"Power (P): {P:.2f} W")
    print(f"Thrust Coefficient (C_T): {C_T:.4f}")
    print(f"Power Coefficient (C_P): {C_P:.4f}")
