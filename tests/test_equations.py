from WindTurbineModeling.equations import *
from pandas import DataFrame

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

