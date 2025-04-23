from time import time
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

from config import *
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.equations import *
from WindTurbineModeling.plot import *

# Start execution timer
start = time()

# ------------------------ Data Class for Boundary Conditions ------------------------
@dataclass
class BoundaryConditions:
    dr: float
    r: float
    Num_Blades: int
    r_s: np.ndarray
    V_0: pd.Series
    theta_p: pd.Series
    omega: pd.Series
    BlSpn: pd.Series
    BlTwist: pd.Series
    BlChord: pd.Series
    BlAFID: pd.Series
    a0: pd.Series
    a0_prime: pd.Series

# ------------------------ Load Input Data ------------------------
# Blade geometry
df_blade_input_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
df_blade_input_data = df_blade_input_data[df_blade_input_data["BlAFID"] > 1].reset_index(drop=True)

# Operational conditions
df_settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)
df_settings = df_settings[(df_settings["WindSpeed"] >= 3.0) & (df_settings["WindSpeed"] <= 25.0)]

# Airfoil definitions
airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])
airfoil_shapes = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
airfoil_coeffs = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]
df_shapes = load_geometry(airfoil_shapes)
_, dfs_airfoil_raw = load_airfoil_coefficients(airfoil_coeffs)

# Match airfoils to IDs
valid_blafids, dfs_airfoil = [], []
for i, df in enumerate(dfs_airfoil_raw):
    valid_blafids.append(i + 1)
    dfs_airfoil.append(df)

# Blade span inputs
r_s = df_blade_input_data['BlSpn'].values
BlSpn = df_blade_input_data['BlSpn']
BlTwist = df_blade_input_data['BlTwist']
BlChord = df_blade_input_data['BlChord']
BlAFID = df_blade_input_data['BlAFID']

# ------------------------ Run Blade Element Momentum Method ------------------------
results = []

for _, row in df_settings.iterrows():
    V0 = row['WindSpeed']
    pitch = row['PitchAngle']
    omega = calc_omega(row['RotSpeed'])  # now using the helper function

    dT_list, dM_list, r_used = [], [], []

    for r, af_id in zip(r_s, BlAFID):
        # Get local blade geometry
        beta, chord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)
        df_polar = dfs_airfoil[valid_blafids.index(af_id)]

        # Initialize induction factors
        a, a_prime = 0.0, 0.0

        for _ in range(MAX_ITER):
            # Flow angle and AoA
            phi = calc_flow_angle(V0, omega, r, a, a_prime)
            alpha = calc_local_angle_of_attack(phi, pitch, beta)

            # Lift and drag
            Cl, Cd = calc_local_lift_drag_force(alpha, df_polar)
            Cn, Ct = calc_normal_tangential_constants(phi, Cl, Cd)

            # Local solidity and Prandtl tip loss
            sigma = calc_local_solidity(NUMBER_BLADES, chord, r)
            F = calc_prandtl_tip_loss(NUMBER_BLADES, R, r, phi)

            # Update induction factors
            a_new, a_prime_new = update_induction_factors(phi, sigma, Cn, Ct, F, a)

            # Check convergence
            if abs(a_new - a) < TOLERANCE and abs(a_prime_new - a_prime) < TOLERANCE:
                break

            a, a_prime = a_new, a_prime_new

        # Compute relative wind velocity and loads
        v_rel, p_n, p_t = calc_relative_velocity_and_forces(V0, omega, r, a, a_prime, RHO, chord, Cn, Ct)

        # Skip if invalid (stall or numerics)
        if np.isnan(Cl) or np.isnan(Cd) or np.isnan(p_t) or Cl == 0 or p_t < 0:
            continue

        # Accumulate local thrust and torque
        dT_list.append(NUMBER_BLADES * p_n)
        dM_list.append(NUMBER_BLADES * r * p_t)
        r_used.append(r)

    # Total thrust, torque, power, coefficients
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

# ------------------------ Output Summary ------------------------
df = pd.DataFrame(results).sort_values("V_0")
print("Final Summary Table:")
print(df[["V_0", "P", "T", "M", "C_P", "C_T"]].round(2).to_string(index=False))

# ------------------------ Plotting ------------------------
V_arr = df["V_0"].values
P_arr = df["P"].values/1e6 # Convert to MW
Cp_arr = df["C_P"].values
T_arr = df["T"].values
Ct_arr = df["C_T"].values

plot_power_curve(V_arr, P_arr, RATED_POWER)  # Convert to MW
plot_cp_curve(V_arr, Cp_arr)
plot_ct_curve(V_arr, Ct_arr)
plot_thrust_curve(V_arr, T_arr)
plot_airfoil_shapes(df_shapes)
plot_wind_turbine()
#power_curve = performance_data.get_power_curve()
#thrust_curve = performance_data.get_thrust_curve()

plt.show()
print(f"Execution time: {time() - start:.2f} s")

