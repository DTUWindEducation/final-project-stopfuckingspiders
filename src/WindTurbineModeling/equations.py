import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from WindTurbineModeling.load import load_operational_settings
from typing import Tuple, Callable, Optional

# TODO: add docstrings to all functions

def calc_omega(rot_speed_rpm: float) -> float:
    """
    Calculate the angular velocity (omega) in radians per second.

    Parameters:
    -----------
    rot_speed_rpm : float
        The rotational speed in revolutions per minute (RPM).

    Returns:
    --------
    float
        The angular velocity in radians per second.
    """
    return rot_speed_rpm * (2 * np.pi / 60)


def calc_flow_angle(V_0: float, omega: float, r: float, a: float, a_prime: float) -> float:
    """
    Calculate the flow angle for a wind turbine blade element.

    Parameters:
        V_0 (float): Free-stream wind velocity (m/s).
        omega (float): Angular velocity of the turbine (rad/s).
        r (float): Radial position on the blade (m).
        a (float): Axial induction factor (dimensionless).
        a_prime (float): Tangential induction factor (dimensionless).

    Returns:
        float: The flow angle (radians).
    """
    return np.arctan(((1 - a) * V_0) / ((1 + a_prime) * omega * r))


def calc_local_angle_of_attack(phi_rad: float, theta_p_deg: float, beta_deg: float) -> float:
    """
    Calculates the local angle of attack for a wind turbine blade.

    Parameters:
        phi_rad (float): The flow angle in radians.
        theta_p_deg (float): The pitch angle of the blade in degrees.
        beta_deg (float): The twist angle of the blade in degrees.

    Returns:
        float: The local angle of attack in degrees.
    """
    return np.degrees(phi_rad) - (theta_p_deg + beta_deg)


def interpolate_blade_geometry(r: float, BlSpn: np.ndarray, BlTwist: np.ndarray, BlChord: np.ndarray) -> Tuple[float, float]:
    """
    Interpolates the blade geometry parameters (twist angle and chord length) at a given radial position.

    Parameters:
        r (float): The radial position along the blade where interpolation is performed.
        BlSpn (np.ndarray): Array of radial positions (spanwise locations) along the blade.
        BlTwist (np.ndarray): Array of twist angles corresponding to the radial positions in BlSpn.
        BlChord (np.ndarray): Array of chord lengths corresponding to the radial positions in BlSpn.

    Returns:
        Tuple[float, float]: A tuple containing:
            - beta (float): The interpolated twist angle at the given radial position.
            - chord (float): The interpolated chord length at the given radial position.
    """
    beta = np.interp(r, BlSpn, BlTwist)
    chord = np.interp(r, BlSpn, BlChord)
    return beta, chord


def calc_local_lift_drag_force(alpha: float, df_polar: dict) -> Tuple[float, float]:
    """
    Calculate the local lift and drag coefficients for a given angle of attack.

    This function interpolates the lift coefficient (Cl) and drag coefficient (Cd)
    from the provided polar data dictionary based on the given angle of attack (alpha).

    Parameters:
        alpha (float): The angle of attack in degrees.
        df_polar (dict): A dictionary containing polar data with the following keys:
            - "Alpha (deg)": List or array of angles of attack in degrees.
            - "Cl": List or array of lift coefficients corresponding to the angles of attack.
            - "Cd": List or array of drag coefficients corresponding to the angles of attack.

    Returns:
        Tuple[float, float]: A tuple containing:
            - Cl (float): The interpolated lift coefficient.
            - Cd (float): The interpolated drag coefficient.
    """
    Cl = np.interp(alpha, df_polar["Alpha (deg)"], df_polar["Cl"])
    Cd = np.interp(alpha, df_polar["Alpha (deg)"], df_polar["Cd"])
    return Cl, Cd


def calc_normal_tangential_constants(phi_rad: float, C_l: float, C_d: float) -> Tuple[float, float]:
    """
    Calculate the normal and tangential force coefficients for a wind turbine blade element.

    Parameters:
    ----------
    phi_rad : float
        The angle of attack in radians.
    C_l : float
        The lift coefficient.
    C_d : float
        The drag coefficient.

    Returns:
    -------
    Tuple[float, float]
        A tuple containing:
        - Cn (float): The normal force coefficient.
        - Ct (float): The tangential force coefficient.
    """
    Cn = C_l * np.cos(phi_rad) + C_d * np.sin(phi_rad)
    Ct = C_l * np.sin(phi_rad) - C_d * np.cos(phi_rad)
    return Cn, Ct


def calc_local_solidity(NUMBER_BLADES: int, chord: float, r: float) -> float:
    """
    Calculate the local solidity of a wind turbine blade section.

    Parameters:
    ----------
    NUMBER_BLADES : int
        The number of blades on the wind turbine.
    chord : float
        The chord length of the blade section (in meters).
    r : float
        The radial distance from the center of the rotor to the blade section (in meters).

    Returns:
    -------
    float
        The local solidity at the given radial position.
    """
    return NUMBER_BLADES * chord / (2 * np.pi * r)


def calc_prandtl_tip_loss(B: int, R: float, r: float, phi: float) -> float:
    """
    This function calculates the correction factor for the loss of lift 
    at the blade tips of a wind turbine, based on Prandtl's tip loss model. 
    The correction factor is used in blade element momentum (BEM) theory 
    to account for the finite number of blades and the effects of tip vortices.

    Parameters:
    -----------
    B : int
        Number of blades on the wind turbine rotor.
    R : float
        Rotor radius (in meters).
    r : float
        Radial position along the blade (in meters).
    phi : float
        Flow angle (in radians).
    Returns:
    --------
    float
        Prandtl's tip loss correction factor, a value between 0 and 1.
        A value closer to 1 indicates minimal tip loss, while a value closer 
        to 0 indicates significant tip loss.
    """
    sin_phi = np.sin(phi)
    
    # Avoid divide-by-zero
    if r <= 0 or sin_phi <= 0:
        return 1e-3  # minimal correction, avoid NaNs

    f_tip = (B / 2.0) * (R - r) / (r * sin_phi)
    
    # Now compute the exponential term safely
    exp_term = np.exp(-f_tip)

    # Ensure exp_term is in domain of arccos
    exp_term = np.clip(exp_term, 0.0, 1.0)

    F = (2.0 / np.pi) * np.arccos(exp_term)

    return np.clip(F, 1e-3, 1.0)


def update_induction_factors(phi_rad: float, sigma: float, C_n: float, C_t: float, F: float, a_old: float, corr: float = 0.1, eps: float = 1e-6) -> Tuple[float, float]:
    """
    Update the axial and tangential induction factors for a wind turbine blade element.
    This function calculates the new axial (`a_new`) and tangential (`a_prime_new`) induction factors
    based on the blade element's angle of attack, solidity, aerodynamic coefficients, and other parameters.
    The calculation uses a relaxation factor (`corr`) to ensure numerical stability.

    Parameters:
        phi_rad (float): Angle of attack in radians.
        sigma (float): Local solidity of the blade element (ratio of blade area to swept area).
        C_n (float): Normal force coefficient.
        C_t (float): Tangential force coefficient.
        F (float): Prandtl's tip loss factor.
        a_old (float): Previous axial induction factor.
        corr (float, optional): Relaxation factor for updating the induction factors. Default is 0.1.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-6.
    Returns:
        Tuple[float, float]: A tuple containing:
            - a_new (float): Updated axial induction factor.
            - a_prime_new (float): Updated tangential induction factor.
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
    a_prime_new = corr * a_prime_star + (1 - corr) * a_prime_star

    return a_new, a_prime_new


def calc_relative_velocity_and_forces(V_0: float, omega: float, r: float, a: float, a_prime: float, RHO: float, chord: float, Cn: float, Ct: float) -> Tuple[float, float, float]:
    """
    Calculate the relative velocity and aerodynamic forces on a wind turbine blade element.

    Parameters:
        V_0 (float): Free-stream wind velocity (m/s).
        omega (float): Rotational speed of the turbine (rad/s).
        r (float): Radial position of the blade element (m).
        a (float): Axial induction factor (dimensionless).
        a_prime (float): Tangential induction factor (dimensionless).
        RHO (float): Air density (kg/m^3).
        chord (float): Chord length of the blade element (m).
        Cn (float): Normal force coefficient (dimensionless).
        Ct (float): Tangential force coefficient (dimensionless).

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - v_rel (float): Relative velocity at the blade element (m/s).
            - p_n (float): Normal force per unit length on the blade element (N/m).
            - p_t (float): Tangential force per unit length on the blade element (N/m).
    """
    tangential = omega * r * (1 + a_prime)
    axial = V_0 * (1 - a)
    v_rel = np.sqrt(tangential**2 + axial**2)
    q = 0.5 * RHO * v_rel**2
    p_n = q * chord * Cn
    p_t = q * chord * Ct
    return v_rel, p_n, p_t


def compute_local_thrust(r_series: np.ndarray, V_0: float, a_series: np.ndarray, RHO: float, dr: float) -> np.ndarray:
    """
    Compute the local thrust distribution along the blade of a wind turbine.

    Parameters:
    -----------
    r_series : np.ndarray
        Array of radial positions along the blade (in meters).
    V_0 : float
        Free-stream wind velocity (in meters per second).
    a_series : np.ndarray
        Array of axial induction factors at each radial position.
    RHO : float
        Air density (in kilograms per cubic meter).
    dr : float
        Radial segment length (in meters).

    Returns:
    --------
    np.ndarray
        Array of local thrust values (in Newtons) for each radial segment.
    """
    return 4 * np.pi * r_series * RHO * V_0**2 * a_series * (1 - a_series) * dr


def compute_rotor_coefficients(T: float, P: float, RHO: float, R: float, V_0: float) -> Tuple[float, float]:
    """
    Compute the rotor power coefficient (CP) and thrust coefficient (CT) for a wind turbine.

    Parameters:
        T (float): Thrust force on the rotor (in Newtons).
        P (float): Mechanical power output of the turbine (in Watts).
        RHO (float): Air density (in kg/m^3).
        R (float): Rotor radius (in meters).
        V_0 (float): Free-stream wind velocity (in m/s).

    Returns:
        Tuple[float, float]: A tuple containing:
            - CP (float): Power coefficient of the rotor.
            - CT (float): Thrust coefficient of the rotor.
    """
    A = np.pi * R**2
    P_wind = 0.5 * RHO * A * V_0**3
    CP = P / P_wind
    CT = T / (0.5 * RHO * A * V_0**2)
    return CP, CT


def compute_totals_and_coefficients(dT: np.ndarray, dM: np.ndarray, omega: float, r_used: np.ndarray, RHO: float, R: float, V_0: float, RATED_POWER: float) -> Tuple[float, float, float, float, float]:
    """
    Compute the total thrust, torque, power, and aerodynamic coefficients for a wind turbine.
    This function integrates the distributed thrust and torque over the blade span to calculate
    the total thrust (T) and torque (M). It then computes the total power (P_total) generated by
    the turbine, ensuring it does not exceed the rated power. Additionally, it calculates the
    power coefficient (CP) and thrust coefficient (CT) based on the wind turbine's performance.
    
    Parameters:
        dT (np.ndarray): Distributed thrust values along the blade span.
        dM (np.ndarray): Distributed torque values along the blade span.
        omega (float): Angular velocity of the turbine (rad/s).
        r_used (np.ndarray): Radial positions along the blade span where dT and dM are defined.
        RHO (float): Air density (kg/m^3).
        R (float): Rotor radius (m).
        V_0 (float): Free-stream wind velocity (m/s).
        RATED_POWER (float): Rated power of the wind turbine (W).
    Returns:
        Tuple[float, float, float, float, float]:
            - T (float): Total thrust (N).
            - M (float): Total torque (Nm).
            - P_total (float): Total power generated by the turbine (W), capped at the rated power.
            - CP (float): Power coefficient, a measure of turbine efficiency.
            - CT (float): Thrust coefficient, a measure of aerodynamic loading.
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


def calculate_optimal_strategy(operational_data: Optional[np.ndarray] = None) -> Tuple[Callable[[float], float], Callable[[float], float], float, float]:
    """
    Calculate the optimal pitch and RPM strategies for a wind turbine based on operational data.
    This function processes operational data to generate interpolation functions for pitch and RPM
    as a function of wind speed. If no data is provided, it loads default operational settings.

    Parameters:
        operational_data (Optional[np.ndarray]): A 2D numpy array where each row represents a data point
            with columns corresponding to wind speed, pitch, and RPM, respectively. If None, default
            operational settings will be loaded.
    Returns:
        Tuple[Callable[[float], float], Callable[[float], float], float, float]:
            - pitch_fn (Callable[[float], float]): Interpolation function for pitch as a function of wind speed.
            - rpm_fn (Callable[[float], float]): Interpolation function for RPM as a function of wind speed.
            - min_wind_speed (float): The minimum wind speed in the operational data.
            - max_wind_speed (float): The maximum wind speed in the operational data.
    """
    # Load data if not provided
    if operational_data is None:
        operational_data = load_operational_settings()
    
    # Extract columns
    wind_speeds = operational_data[:, 0]
    pitch = operational_data[:, 1]
    rpm = operational_data[:, 2]
    
    # Process pitch curve
    pitch_smoothed = savgol_filter(pitch, window_length=5, polyorder=2)
    
    # Create interpolation functions
    pitch_fn = interp1d(
        wind_speeds, pitch_smoothed,
        kind='linear',
        bounds_error=False,
        fill_value=(pitch_smoothed[0], pitch_smoothed[-1])
    )
    
    rpm_fn = interp1d(
        wind_speeds, rpm,
        kind='linear',
        bounds_error=False,
        fill_value=(rpm[0], rpm[-1])
    )
    
    return pitch_fn, rpm_fn, wind_speeds.min(), wind_speeds.max()