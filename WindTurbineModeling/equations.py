import numpy as np

def calc_omega(rot_speed_rpm):
    """
    Convert rotational speed from RPM to angular velocity in radians per second.

    Parameters
    ----------
    rot_speed_rpm : float
        Rotational speed [RPM].

    Returns
    -------
    float
        Angular velocity [rad/s].
    """
    return rot_speed_rpm * (2 * np.pi / 60)

def calc_flow_angle(V_0, omega, r, a, a_prime):
    """
    Compute the local flow angle φ (phi) in radians.

    Parameters
    ----------
    V_0 : float
        Freestream wind speed [m/s].
    omega : float
        Rotor angular velocity [rad/s].
    r : float
        Radial location [m].
    a : float
        Axial induction factor.
    a_prime : float
        Tangential induction factor.

    Returns
    -------
    float
        Flow angle φ [rad].
    """
    return np.arctan(((1 - a) * V_0) / ((1 + a_prime) * omega * r))


def calc_local_angle_of_attack(phi_rad, theta_p_deg, beta_deg):
    """
    Calculate the angle of attack in degrees.

    Parameters
    ----------
    phi_rad : float
        Flow angle [rad].
    theta_p_deg : float
        Pitch angle [deg].
    beta_deg : float
        Local twist angle [deg].

    Returns
    -------
    float
        Angle of attack [deg].
    """
    return np.degrees(phi_rad) - (theta_p_deg + beta_deg)


def interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord):
    """
    Linearly interpolate the blade twist and chord at a given spanwise location.

    Parameters
    ----------
    r : float
        Radial location [m].
    BlSpn : array-like
        Spanwise positions of the blade [m].
    BlTwist : array-like
        Twist angles along the span [deg].
    BlChord : array-like
        Chord lengths along the span [m].

    Returns
    -------
    tuple (float, float)
        Interpolated (beta, chord) at r.
    """
    beta = np.interp(r, BlSpn, BlTwist)
    chord = np.interp(r, BlSpn, BlChord)
    return beta, chord


def calc_local_lift_drag_force(alpha, df_polar):
    """
    Interpolate lift and drag coefficients from airfoil polar data.

    Parameters
    ----------
    alpha : float
        Angle of attack [deg].
    df_polar : pd.DataFrame
        Airfoil polar data containing "Alpha (deg)", "Cl", and "Cd".

    Returns
    -------
    tuple (float, float)
        Interpolated lift (Cl) and drag (Cd) coefficients.
    """
    Cl = np.interp(alpha, df_polar["Alpha (deg)"], df_polar["Cl"])
    Cd = np.interp(alpha, df_polar["Alpha (deg)"], df_polar["Cd"])
    return Cl, Cd


def calc_normal_tangential_constants(phi_rad, C_l, C_d):
    """
    Compute normal (Cn) and tangential (Ct) force coefficients.

    Parameters
    ----------
    phi_rad : float
        Flow angle [rad].
    C_l : float
        Lift coefficient.
    C_d : float
        Drag coefficient.

    Returns
    -------
    tuple (float, float)
        Normal and tangential force coefficients (Cn, Ct).
    """
    Cn = C_l * np.cos(phi_rad) + C_d * np.sin(phi_rad)
    Ct = C_l * np.sin(phi_rad) - C_d * np.cos(phi_rad)
    return Cn, Ct


def calc_local_solidity(NUMBER_BLADES, chord, r):
    """
    Compute the local solidity of the blade element.

    Parameters
    ----------
    NUMBER_BLADES : int
        Number of rotor blades.
    chord : float
        Local chord length [m].
    r : float
        Radial location [m].

    Returns
    -------
    float
        Local solidity σ.
    """
    return NUMBER_BLADES * chord / (2 * np.pi * r)


def calc_prandtl_tip_loss(B, R, r, phi):
    """
    Compute Prandtl's tip loss correction factor.

    Parameters
    ----------
    B : int
        Number of blades.
    R : float
        Rotor radius [m].
    r : float
        Local radial position [m].
    phi : float
        Flow angle [rad].

    Returns
    -------
    float
        Tip loss factor F ∈ (0, 1].
    """
    f_tip = (B / 2) * (R - r) / (r * np.sin(phi) + 1e-6)
    F = (2 / np.pi) * np.arccos(np.exp(-f_tip))
    return np.clip(F, 1e-3, 1.0)


def update_induction_factors(phi_rad, sigma, C_n, C_t, F, a_old, corr=0.1, eps=1e-6):
    """
    Update axial and tangential induction factors using Glauert’s correction.

    Parameters
    ----------
    phi_rad : float
        Flow angle [rad].
    sigma : float
        Local solidity.
    C_n : float
        Normal force coefficient.
    C_t : float
        Tangential force coefficient.
    F : float
        Tip loss factor.
    a_old : float
        Previous value of axial induction factor.
    corr : float, optional
        Relaxation factor for iteration. Default is 0.1.
    eps : float, optional
        Small number to avoid division by zero. Default is 1e-6.

    Returns
    -------
    tuple (float, float)
        Updated axial and tangential induction factors (a, a_prime).
    """
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    sin_phi_sq = sin_phi ** 2

    if a_old <= 1/3:
        a_star = (1 - a_old) * sigma * C_n / (4 * F * sin_phi_sq + eps)
    else:
        CT = ((1 - a_old)**2) * C_n * sigma / (sin_phi_sq + eps)
        a_star = CT / (4 * F * (1 - 0.25 * (5 - 3 * a_old) * a_old) + eps)

    a_prime_star = (1 - a_old) * sigma * C_t / (4 * F * sin_phi * cos_phi + eps)

    a_new = corr * a_star + (1 - corr) * a_old
    a_prime_new = corr * a_prime_star + (1 - corr) * a_prime_star  # assumes no history for a_prime

    return a_new, a_prime_new

def calc_relative_velocity_and_forces(V_0, omega, r, a, a_prime, RHO, chord, Cn, Ct):
    """
    Compute relative wind velocity and aerodynamic forces (normal and tangential)
    at a blade element.

    Parameters
    ----------
    V_0 : float
        Freestream wind speed [m/s].
    omega : float
        Angular velocity of the rotor [rad/s].
    r : float
        Radial position of the blade element [m].
    a : float
        Axial induction factor.
    a_prime : float
        Tangential induction factor.
    RHO : float
        Air density [kg/m^3].
    chord : float
        Local chord length [m].
    Cn : float
        Normal force coefficient.
    Ct : float
        Tangential force coefficient.

    Returns
    -------
    tuple
        (v_rel, p_n, p_t)
        v_rel : float
            Relative wind speed at the blade element [m/s].
        p_n : float
            Normal (thrust) force per unit span [N/m].
        p_t : float
            Tangential (torque) force per unit span [N/m].
    """
    tangential = omega * r * (1 + a_prime)
    axial = V_0 * (1 - a)
    v_rel = np.sqrt(tangential**2 + axial**2)

    q = 0.5 * RHO * v_rel**2  # dynamic pressure
    p_n = q * chord * Cn
    p_t = q * chord * Ct

    return v_rel, p_n, p_t

def compute_local_thrust(r_series, V_0, a_series, RHO, dr):
    """
    Compute local thrust across blade span (differential form).

    Parameters
    ----------
    r_series : array-like
        Radial positions [m].
    V_0 : float
        Freestream wind speed [m/s].
    a_series : array-like
        Axial induction factors along blade.
    RHO : float
        Air density [kg/m^3].
    dr : float
        Radial differential length [m].

    Returns
    -------
    np.ndarray
        Local thrust forces per element [N].
    """
    return 4 * np.pi * r_series * RHO * V_0**2 * a_series * (1 - a_series) * dr


def compute_rotor_coefficients(T, P, RHO, R, V_0):
    """
    Compute power and thrust coefficients (Cp, Ct) from rotor outputs.

    Parameters
    ----------
    T : float
        Total thrust [N].
    P : float
        Total power [W].
    RHO : float
        Air density [kg/m^3].
    R : float
        Rotor radius [m].
    V_0 : float
        Freestream wind speed [m/s].

    Returns
    -------
    tuple (float, float)
        Power coefficient Cp, thrust coefficient Ct.
    """
    A = np.pi * R**2
    P_wind = 0.5 * RHO * A * V_0**3
    CP = P / P_wind
    CT = T / (0.5 * RHO * A * V_0**2)
    return CP, CT


def compute_totals_and_coefficients(dT, dM, omega, r_used, RHO, R, V_0, RATED_POWER):
    """
    Integrate elemental thrust and torque to compute total performance metrics.

    Parameters
    ----------
    dT : list
        Elemental thrust forces [N].
    dM : list
        Elemental torque values [Nm].
    omega : float
        Rotor speed [rad/s].
    r_used : list
        Radial positions [m].
    RHO : float
        Air density [kg/m^3].
    R : float
        Rotor radius [m].
    V_0 : float
        Freestream wind speed [m/s].
    RATED_POWER : float
        Maximum rated output power [W].

    Returns
    -------
    tuple (T, M, P_total, CP, CT)
        Total thrust, torque, power, and non-dimensional performance coefficients.
    """
    T = np.trapz(dT, x=r_used)
    M = np.trapz(dM, x=r_used)
    P_total = M * omega
    P_wind = 0.5 * RHO * np.pi * R**2 * V_0**3

    if P_total > RATED_POWER:
        P_total = RATED_POWER

    CP = P_total / P_wind
    CT = T / (0.5 * RHO * np.pi * R**2 * V_0**2)

    return T, M, P_total, CP, CT

