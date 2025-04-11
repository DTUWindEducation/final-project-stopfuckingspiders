import numpy as np
import pandas as pd

def calc_flow_angle(bc):
    """
    Calculate the flow angle (phi) for a blade element.

    Parameters:
        bc (BoundaryConditions): Object containing boundary conditions.

    Returns:
        phi (float or pd.Series): Flow angle [rad].
    """
    a = bc.a0
    a_prime = bc.a0_prime
    V_0 = bc.V_0
    omega = bc.omega
    r = bc.r
    phi = np.arctan((1 - a) * V_0 / ((1 + a_prime) * omega * r))
    return phi

def calc_local_angle_of_attack(phi, theta_p, beta):
    """
    Calculate the local angle of attack (alpha) from flow angle, pitch, and twist.

    Parameters:
        phi (float or pd.Series): Flow angle [rad].
        theta_p (pd.Series): Blade pitch angle [deg].
        beta (float): Local twist angle [deg].

    Returns:
        alpha (pd.Series): Local angle of attack [deg].
    """
    return phi - (theta_p + beta)

def calc_local_lift_drag_force(alpha, df):
    """
    Interpolate lift and drag coefficients from airfoil data.

    Parameters:
        alpha (pd.Series): Local angle of attack [deg].
        df (pd.DataFrame): Airfoil coefficient table with columns 'Alpha (deg)', 'Cl', 'Cd'.

    Returns:
        loc_C_d (pd.Series): Local drag coefficient [-].
        loc_C_l (pd.Series): Local lift coefficient [-].
    """
    loc_C_d = pd.Series(np.interp(alpha, df['Alpha (deg)'], df['Cd']), name="C_d [-]")
    loc_C_l = pd.Series(np.interp(alpha, df['Alpha (deg)'], df['Cl']), name="C_l [-]")
    return loc_C_d, loc_C_l

def calc_local_solidity(bc, loc_BlChord):
    """
    Compute local solidity (sigma) of the blade element.

    Parameters:
        bc (BoundaryConditions): Object containing blade/radius properties.
        loc_BlChord (float): Local chord length [m].

    Returns:
        sigma (float): Local solidity [-].
    """
    return (loc_BlChord * bc.Num_Blades) / (2 * np.pi * bc.r)

def calc_normal_tangential_constants(phi, C_d, C_l):
    """
    Compute normal and tangential aerodynamic force coefficients.

    Parameters:
        phi (float or pd.Series): Flow angle [rad].
        C_d (pd.Series): Drag coefficient [-].
        C_l (pd.Series): Lift coefficient [-].

    Returns:
        C_n (pd.Series): Normal force coefficient [-].
        C_t (pd.Series): Tangential force coefficient [-].
    """
    C_n = C_l * np.cos(phi) + C_d * np.sin(phi)
    C_t = C_l * np.sin(phi) + C_d * np.cos(phi)
    return C_n, C_t

def update_induction_factors(phi, sigma, C_n, C_t):
    """
    Compute updated axial and tangential induction factors.

    Parameters:
        phi (pd.Series): Flow angle [rad].
        sigma (float): Local solidity [-].
        C_n (pd.Series): Normal force coefficient [-].
        C_t (pd.Series): Tangential force coefficient [-].

    Returns:
        a (pd.Series): Axial induction factor [-].
        a_prime (pd.Series): Tangential induction factor [-].
    """
    a = 1 / ((4 * np.sin(phi)**2) / (sigma * C_n + 1))
    a_prime = 1 / ((4 * np.sin(phi) * np.cos(phi)) / (sigma * C_t - 1))
    return a, a_prime

def compute_local_thrust(r_series, V_0, a, RHO, dr):
    """
    Compute differential thrust dT along blade span.

    Parameters:
        r_series (pd.Series): Spanwise position [m].
        V_0 (pd.Series): Inflow wind speed [m/s].
        a (pd.Series): Axial induction factor [-].
        RHO (float): Air density [kg/m³].
        dr (float): Radial increment [m].

    Returns:
        dT (pd.Series): Local thrust [N].
    """
    return 4 * np.pi * r_series * RHO * V_0**2 * a * (1 - a) * dr

def compute_local_torque(r_series, V_0, a, a_prime, omega, RHO, dr):
    """
    Compute differential torque dM along blade span.

    Parameters:
        r_series (pd.Series): Spanwise position [m].
        V_0 (pd.Series): Inflow wind speed [m/s].
        a (pd.Series): Axial induction factor [-].
        a_prime (pd.Series): Tangential induction factor [-].
        omega (pd.Series): Rotational speed [rad/s].
        RHO (float): Air density [kg/m³].
        dr (float): Radial increment [m].

    Returns:
        dM (pd.Series): Local torque [Nm].
    """
    return 4 * np.pi * r_series**3 * RHO * V_0 * omega * a_prime * (1 - a) * dr

def interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord):
    """
    Interpolate blade twist and chord length at a specific radial location.

    Parameters:
        r (float): Target radius location [m].
        BlSpn (pd.Series): Blade spanwise node locations [m].
        BlTwist (pd.Series): Twist distribution [deg].
        BlChord (pd.Series): Chord length distribution [m].

    Returns:
        twist (float): Interpolated twist [deg].
        chord (float): Interpolated chord [m].
    """
    twist = np.interp(r, BlSpn, BlTwist)
    chord = np.interp(r, BlSpn, BlChord)
    return twist, chord

def compute_rotor_coefficients(T, P, RHO, R, V_0):
    """
    Compute non-dimensional thrust and power coefficients.

    Parameters:
        T (float): Total thrust [N].
        P (float): Total power [W].
        RHO (float): Air density [kg/m³].
        R (float): Rotor radius [m].
        V_0 (pd.Series): Inflow wind speed [m/s].

    Returns:
        C_T (float): Thrust coefficient [-].
        C_P (float): Power coefficient [-].
    """
    A = np.pi * R**2
    V0_mean = V_0.mean()
    C_T = T / (0.5 * RHO * A * V0_mean**2)
    C_P = P / (0.5 * RHO * A * V0_mean**3)
    return C_T, C_P

def compute_totals_and_coefficients(dT, dM, omega, dr, RHO, R, V_0):
    """
    Compute total thrust, torque, power, and their coefficients.

    Parameters:
        dT (pd.Series): Local thrust [N].
        dM (pd.Series): Local torque [Nm].
        omega (pd.Series): Rotational speed [rad/s].
        dr (float): Radial increment [m].
        RHO (float): Air density [kg/m³].
        R (float): Rotor radius [m].
        V_0 (pd.Series): Inflow wind speed [m/s].

    Returns:
        T (float): Total thrust [N].
        M (float): Total torque [Nm].
        P (float): Total power [W].
        C_T (float): Thrust coefficient [-].
        C_P (float): Power coefficient [-].
    """
    T = np.trapz(dT, dx=dr)
    M = np.trapz(dM, dx=dr)
    P = np.trapz(omega * dM / dr, dx=dr)
    C_T, C_P = compute_rotor_coefficients(T, P, RHO, R, V_0)
    return T, M, P, C_T, C_P

def print_summary(T, M, P, C_T, C_P):
    """
    Print performance metrics for the rotor.

    Parameters:
        T (float): Thrust [N].
        M (float): Torque [Nm].
        P (float): Power [W].
        C_T (float): Thrust coefficient [-].
        C_P (float): Power coefficient [-].
    """
    print(f"Thrust (T): {T:.2f} N")
    print(f"Torque (M): {M:.2f} Nm")
    print(f"Power (P): {P:.2f} W")
    print(f"Thrust Coefficient (C_T): {C_T:.4f}")
    print(f"Power Coefficient (C_P): {C_P:.4f}")
