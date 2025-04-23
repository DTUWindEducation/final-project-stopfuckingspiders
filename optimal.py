from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from config import *
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.equations import *
from WindTurbineModeling.plot import *

# ------------------------ Run BEM Function ------------------------
def run_bem_loop(V0_array, pitch_fn, rpm_fn, r_s, BlAFID, BlSpn, BlTwist, BlChord, dfs_airfoil, valid_blafids):
    results = []

    for V0 in V0_array:
        pitch = float(pitch_fn(V0))
        rpm = float(rpm_fn(V0))
        omega = calc_omega(rpm)

        dT_list, dM_list, r_used = [], [], []

        for r, af_id in zip(r_s, BlAFID):
            beta, chord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)
            df_polar = dfs_airfoil[valid_blafids.index(af_id)]

            a, a_prime = 0.0, 0.0
            for _ in range(MAX_ITER):
                phi = calc_flow_angle(V0, omega, r, a, a_prime)
                alpha = calc_local_angle_of_attack(phi, pitch, beta)
                Cl, Cd = calc_local_lift_drag_force(alpha, df_polar)
                Cn, Ct = calc_normal_tangential_constants(phi, Cl, Cd)
                sigma = calc_local_solidity(NUMBER_BLADES, chord, r)
                F = calc_prandtl_tip_loss(NUMBER_BLADES, R, r, phi)

                a_new, a_prime_new = update_induction_factors(phi, sigma, Cn, Ct, F, a)
                if abs(a_new - a) < TOLERANCE and abs(a_prime_new - a_prime) < TOLERANCE:
                    break
                a, a_prime = a_new, a_prime_new

            v_rel, p_n, p_t = calc_relative_velocity_and_forces(V0, omega, r, a, a_prime, RHO, chord, Cn, Ct)

            if np.isnan(Cl) or np.isnan(Cd) or np.isnan(p_t) or Cl == 0 or p_t < 0:
                continue

            dT_list.append(NUMBER_BLADES * p_n)
            dM_list.append(NUMBER_BLADES * r * p_t)
            r_used.append(r)

        T, M, P_total, CP, CT = compute_totals_and_coefficients(
            dT_list, dM_list, omega, r_used, RHO, R, V0, RATED_POWER
        )

        results.append({
            "V_0": V0,
            "T": T,
            "M": M,
            "P": P_total,
            "C_P": CP,
            "C_T": CT,
        })

    return pd.DataFrame(results).sort_values("V_0")

# ------------------------ Start Execution ------------------------
start = time()

# ------------------------ Load Input Data ------------------------
df_blade_input_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
df_blade_input_data = df_blade_input_data[df_blade_input_data["BlAFID"] > 1].reset_index(drop=True)

df_settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)
df_settings = df_settings[(df_settings["WindSpeed"] >= 3.0) & (df_settings["WindSpeed"] <= 25.0)]

airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])
airfoil_shapes = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
airfoil_coeffs = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]
df_shapes = load_geometry(airfoil_shapes)
_, dfs_airfoil_raw = load_airfoil_coefficients(airfoil_coeffs)

valid_blafids, dfs_airfoil = [], []
for i, df in enumerate(dfs_airfoil_raw):
    valid_blafids.append(i + 1)
    dfs_airfoil.append(df)

r_s = df_blade_input_data['BlSpn'].values
BlSpn = df_blade_input_data['BlSpn']
BlTwist = df_blade_input_data['BlTwist']
BlChord = df_blade_input_data['BlChord']
BlAFID = df_blade_input_data['BlAFID']

# ------------------------ Run Baseline Strategy ------------------------
df_baseline = run_bem_loop(
    df_settings['WindSpeed'].values,
    pitch_fn=lambda V: df_settings[df_settings['WindSpeed'] == V]['PitchAngle'].values[0],
    rpm_fn=lambda V: df_settings[df_settings['WindSpeed'] == V]['RotSpeed'].values[0],
    r_s=r_s,
    BlAFID=BlAFID,
    BlSpn=BlSpn,
    BlTwist=BlTwist,
    BlChord=BlChord,
    dfs_airfoil=dfs_airfoil,
    valid_blafids=valid_blafids
)

# ------------------------ Run Optimal Strategy ------------------------
pitch_interp, rpm_interp, min_wind_speed, max_wind_speed = calculate_optimal_strategy(
    operational_data=df_settings.values
)
wind_speed_range = generate_wind_speed_range(min_wind_speed, max_wind_speed)

df_optimal = run_bem_loop(
    V0_array=wind_speed_range,
    pitch_fn=pitch_interp,
    rpm_fn=rpm_interp,
    r_s=r_s,
    BlAFID=BlAFID,
    BlSpn=BlSpn,
    BlTwist=BlTwist,
    BlChord=BlChord,
    dfs_airfoil=dfs_airfoil,
    valid_blafids=valid_blafids
)


# ------------------------ Plotting ------------------------
V0_array = df_optimal["V_0"].values
P_arr = df_optimal["P"].values/1e6  # Convert to MW
Cp_arr = df_optimal["C_P"].values
T_arr = df_optimal["T"].values
Ct_arr = df_optimal["C_T"].values
# Calculate pitch angles using pitch_fn
pitch_arr = [pitch_interp(V0) for V0 in df_optimal["V_0"].values]
# Similarly for RPM:
rpm_arr = [rpm_interp(V0) for V0 in df_optimal["V_0"].values]
# ------------------------ Plotting ------------------------
# Create first figure for Power and Thrust curves
plt.figure(figsize=(12, 5))

# Power curve (left subplot)
plt.subplot(1, 2, 1)
plt.plot(V0_array, P_arr, 'b-', linewidth=2.5, label='Power Output')
if RATED_POWER:
    plt.axhline(y=RATED_POWER/1e6, color='r', linestyle='--', label='Rated Power')
plt.title('Power Curve', fontsize=12)
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Power [MW]')
plt.grid(True)
plt.legend()

# Thrust curve (right subplot)
plt.subplot(1, 2, 2)
plt.plot(V0_array, T_arr/1000, 'g-', linewidth=2.5, label='Thrust')
plt.title('Thrust Curve', fontsize=12)
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Thrust [kN]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Create second figure for Control Strategies
plt.figure(figsize=(12, 5))

# Pitch angle control (left subplot)
plt.subplot(1, 2, 1)
plt.plot(V0_array, pitch_arr, 'b-', linewidth=2)
plt.title('Pitch Angle Control Strategy', fontsize=12)
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Pitch angle [deg]')
plt.grid(True)

# RPM control (right subplot)
plt.subplot(1, 2, 2)
plt.plot(V0_array, rpm_arr, 'r-', linewidth=2)
plt.title('RPM Control Strategy', fontsize=12)
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Rotational speed [RPM]')
plt.grid(True)

plt.tight_layout()
plt.show()

# Create third figure for Strategy Comparison (keep this separate)
plt.figure(figsize=(10, 6))
plt.plot(df_baseline["V_0"], df_baseline["P"] / 1e6, label="Baseline", marker='o')
plt.plot(df_optimal["V_0"], df_optimal["P"] / 1e6, label="Optimal", marker='x')
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (MW)")
plt.title("Power Curve Comparison")
plt.grid(True)
plt.legend()
plt.show()