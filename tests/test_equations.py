from pandas import DataFrame
import numpy as np

from WindTurbineModeling.equations import *

def test_calc_omega():
    out = calc_omega(1)
    exp_out = 0.10471975511965977
    assert exp_out == out

def test_calc_flow_angle():
    out = calc_flow_angle(1,1,1,2,-2)
    exp_out = 0.7853981633974483
    assert out == exp_out

def test_calc_local_angle_of_attack():
    #10 deg to rad: 0.17453292519943295
    out = calc_local_angle_of_attack(0.17453292519943295,45, 45)
    exp_out = -80
    assert out == exp_out

def test_interpolate_blade_geometry():
    r = 1.5
    x = [1,2,3]
    y = [2,3,4]
    out_beta, out_chord = interpolate_blade_geometry(r, x, y, y)
    exp_out = 2.5
    assert out_beta == exp_out
    assert out_chord == exp_out

def test_calc_local_lift_drag_force():
    r=1.5
    test_df = DataFrame()
    test_df["Alpha (deg)"] = [1, 2, 3]
    test_df["Cl"] = [2,3,4]
    test_df["Cd"] = [2,3,4]
    out_Cn, out_Ct = calc_local_lift_drag_force(r, test_df)
    exp_out = 2.5
    assert out_Cn == exp_out
    assert out_Ct == exp_out

def test_calc_normal_tangential_constants():
    phi = np.radians(45)
    C_l = 2
    C_d = 1
    out_Cn, out_Ct = calc_normal_tangential_constants(phi, C_l, C_d)
    exp_Cn = 2 * np.cos(phi) + 1 * np.sin(phi)
    exp_Ct = 2 * np.sin(phi) - 1 * np.cos(phi)
    assert np.isclose(out_Cn, exp_Cn)
    assert np.isclose(out_Ct, exp_Ct)


def test_calc_local_solidity():
    out = calc_local_solidity(NUMBER_BLADES=3, chord=2, r=5)
    exp_out = 3 * 2 / (2 * np.pi * 5)
    assert np.isclose(out, exp_out)


def test_calc_prandtl_tip_loss_typical():
    out = calc_prandtl_tip_loss(B=3, R=10, r=5, phi=np.radians(30))
    assert 1e-3 <= out <= 1.0


def test_calc_prandtl_tip_loss_zero_radius():
    out = calc_prandtl_tip_loss(B=3, R=10, r=0, phi=np.radians(30))
    assert np.isclose(out, 1e-3)


def test_update_induction_factors_low_a():
    phi = np.radians(20)
    out_a, out_a_prime = update_induction_factors(
        phi_rad=phi, sigma=0.05, C_n=1.2, C_t=0.1, F=0.97, a_old=0.3,
        corr=0.1, eps=1e-6
    )
    assert 0 < out_a < 1
    assert 0 < out_a_prime < 1


def test_calc_relative_velocity_and_forces():
    out_v_rel, out_pn, out_pt = calc_relative_velocity_and_forces(
        V_0=10, omega=2, r=5, a=0.1, a_prime=0.1,
        RHO=1.2, chord=1, Cn=1, Ct=0.5
    )
    tangential = 2 * 5 * 1.1
    axial = 10 * 0.9
    v_rel_exp = np.sqrt(tangential**2 + axial**2)
    q = 0.5 * 1.2 * v_rel_exp**2
    pn_exp = q * 1 * 1
    pt_exp = q * 1 * 0.5
    assert np.isclose(out_v_rel, v_rel_exp)
    assert np.isclose(out_pn, pn_exp)
    assert np.isclose(out_pt, pt_exp)


def test_compute_local_thrust():
    r_series = np.array([1.0])
    a_series = np.array([0.2])
    out = compute_local_thrust(r_series, V_0=10, a_series=a_series, RHO=1.2, dr=1.0)
    exp = 4 * np.pi * 1.0 * 1.2 * 10**2 * 0.2 * (1 - 0.2) * 1.0
    assert np.isclose(out[0], exp)


def test_compute_rotor_coefficients():
    out_cp, out_ct = compute_rotor_coefficients(
        T=1000, P=100000, RHO=1.2, R=5, V_0=10
    )
    A = np.pi * 5**2
    P_wind = 0.5 * 1.2 * A * 10**3
    CT_exp = 1000 / (0.5 * 1.2 * A * 10**2)
    CP_exp = 100000 / P_wind
    assert np.isclose(out_cp, CP_exp)
    assert np.isclose(out_ct, CT_exp)


def test_compute_totals_and_coefficients():
    r_used = np.array([1.0, 2.0])
    dT = np.array([10.0, 20.0])
    dM = np.array([5.0, 15.0])
    out_T, out_M, out_P, out_CP, out_CT = compute_totals_and_coefficients(
        dT=dT, dM=dM, omega=2, r_used=r_used,
        RHO=1.2, R=5, V_0=10, RATED_POWER=1e6
    )
    T_exp = np.trapz(dT, r_used)
    M_exp = np.trapz(dM, r_used)
    P_exp = M_exp * 2
    A = np.pi * 5**2
    CP_exp = P_exp / (0.5 * 1.2 * A * 10**3)
    CT_exp = T_exp / (0.5 * 1.2 * A * 10**2)
    assert np.isclose(out_T, T_exp)
    assert np.isclose(out_M, M_exp)
    assert np.isclose(out_P, P_exp)
    assert np.isclose(out_CP, CP_exp)
    assert np.isclose(out_CT, CT_exp)
