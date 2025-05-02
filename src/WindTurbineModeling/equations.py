import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from WindTurbineModeling.load import load_operational_settings
from typing import Tuple, Callable, Optional

# TODO: add docstrings to all functions

def calc_omega(rot_speed_rpm: float) -> float:
    return rot_speed_rpm * (2 * np.pi / 60)


def calc_flow_angle(V_0: float, omega: float, r: float, a: float, a_prime: float) -> float:
    return np.arctan(((1 - a) * V_0) / ((1 + a_prime) * omega * r))


def calc_local_angle_of_attack(phi_rad: float, theta_p_deg: float, beta_deg: float) -> float:
    return np.degrees(phi_rad) - (theta_p_deg + beta_deg)


def interpolate_blade_geometry(r: float, BlSpn: np.ndarray, BlTwist: np.ndarray, BlChord: np.ndarray) -> Tuple[float, float]:
    beta = np.interp(r, BlSpn, BlTwist)
    chord = np.interp(r, BlSpn, BlChord)
    return beta, chord


def calc_local_lift_drag_force(alpha: float, df_polar: dict) -> Tuple[float, float]:
    Cl = np.interp(alpha, df_polar["Alpha (deg)"], df_polar["Cl"])
    Cd = np.interp(alpha, df_polar["Alpha (deg)"], df_polar["Cd"])
    return Cl, Cd


def calc_normal_tangential_constants(phi_rad: float, C_l: float, C_d: float) -> Tuple[float, float]:
    Cn = C_l * np.cos(phi_rad) + C_d * np.sin(phi_rad)
    Ct = C_l * np.sin(phi_rad) - C_d * np.cos(phi_rad)
    return Cn, Ct


def calc_local_solidity(NUMBER_BLADES: int, chord: float, r: float) -> float:
    return NUMBER_BLADES * chord / (2 * np.pi * r)


def calc_prandtl_tip_loss(B: int, R: float, r: float, phi: float) -> float:
    """
    Compute Prandtl's tip loss correction factor.
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
    tangential = omega * r * (1 + a_prime)
    axial = V_0 * (1 - a)
    v_rel = np.sqrt(tangential**2 + axial**2)
    q = 0.5 * RHO * v_rel**2
    p_n = q * chord * Cn
    p_t = q * chord * Ct
    return v_rel, p_n, p_t


def compute_local_thrust(r_series: np.ndarray, V_0: float, a_series: np.ndarray, RHO: float, dr: float) -> np.ndarray:
    return 4 * np.pi * r_series * RHO * V_0**2 * a_series * (1 - a_series) * dr


def compute_rotor_coefficients(T: float, P: float, RHO: float, R: float, V_0: float) -> Tuple[float, float]:
    A = np.pi * R**2
    P_wind = 0.5 * RHO * A * V_0**3
    CP = P / P_wind
    CT = T / (0.5 * RHO * A * V_0**2)
    return CP, CT


def compute_totals_and_coefficients(dT: np.ndarray, dM: np.ndarray, omega: float, r_used: np.ndarray, RHO: float, R: float, V_0: float, RATED_POWER: float) -> Tuple[float, float, float, float, float]:
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
    Calculates optimal pitch and RPM interpolation functions
    Args:
        operational_data: Optional pre-loaded data array. If None, loads fresh data.
    Returns:
        pitch_fn, rpm_fn, min_wind_speed, max_wind_speed
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