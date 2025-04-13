from time import time
start = time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.equations import *
from WindTurbineModeling.plot import *

from dataclasses import dataclass

#%% -------------------- BoundaryConditions Class --------------------
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
    a0: pd.Series
    a0_prime: pd.Series

#%% -------------------- Load Data --------------------
# Blade and operational inputs
df_blade_input_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
df_settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)

# Cut-in / out wind speed filtering
df_settings = df_settings[(df_settings["WindSpeed"] >= 3.0) & (df_settings["WindSpeed"] <= 25.0)]

# Load airfoil data (geometry + aero)
airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])
airfoil_shapes = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
airfoil_coeffs = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]
df_shapes = load_geometry(airfoil_shapes)
_, df_coeffs = load_airfoil_coefficients(airfoil_coeffs)

# Use first airfoil for now (single-airfoil assumption)
df_airfoil = df_coeffs[0]

#%% -------------------- Plot Airfoils and Turbine --------------------
plot_airfoil_shapes(df_shapes)
plot_wind_turbine(R=R, r=3, dr=DR)

#%% -------------------- Setup Blade Geometry and Span --------------------
dr = DR
r_s = np.arange(dr, R + dr, dr)
BlSpn = df_blade_input_data['BlSpn']
BlTwist = df_blade_input_data['BlTwist']
BlChord = df_blade_input_data['BlChord']

#%% -------------------- Run Simulation --------------------
TOLERANCE = 1e-8
MAX_ITER = 100
all_results = []

# Loop over operational conditions
for _, row in df_settings.iterrows():
    V_0_scalar = row['WindSpeed']
    theta_p_scalar = row['PitchAngle']
    omega_scalar = row['RotSpeed'] * (2 * np.pi / 60)

    BC = BoundaryConditions(
        dr=dr,
        r=0,
        Num_Blades=NUMBER_BLADES,
        r_s=r_s,
        V_0=pd.Series([V_0_scalar]*len(r_s), index=r_s),
        theta_p=pd.Series([theta_p_scalar]*len(r_s), index=r_s),
        omega=pd.Series([omega_scalar]*len(r_s), index=r_s),
        BlSpn=BlSpn,
        BlTwist=BlTwist,
        BlChord=BlChord,
        a0=0.0,
        a0_prime=0.0
    )

    dT_list, dM_list = [], []
    a_list, a_prime_list, alpha_list = [], [], []

    for r in r_s:
        loc_BlTwist, loc_BlChord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)
        BC.r = r
        sigma = calc_local_solidity(BC, loc_BlChord)

        a = 0.0
        a_prime = 0.0

        for _ in range(MAX_ITER):
            BC.a0 = a
            BC.a0_prime = a_prime

            phi = calc_flow_angle(BC)
            alpha = calc_local_angle_of_attack(phi, BC.theta_p, loc_BlTwist)
            C_d, C_l = calc_local_lift_drag_force(alpha, df_airfoil)
            C_n, C_t = calc_normal_tangential_constants(phi, C_d, C_l)
            F = calc_prandtl_tip_loss(NUMBER_BLADES, R, r, phi)

            a_new, a_prime_new = update_induction_factors(phi, sigma, C_n, C_t, F, a)

            a_new = float(np.mean(a_new))
            a_prime_new = float(np.mean(a_prime_new))

            if abs(a_new - a) < TOLERANCE and abs(a_prime_new - a_prime) < TOLERANCE:
                break

            a, a_prime = a_new, a_prime_new

        a_list.append(a)
        a_prime_list.append(a_prime)

        r_series = pd.Series([r], index=[r])
        dT = compute_local_thrust(r_series, V_0_scalar, a, RHO, dr)
        dM = compute_local_torque(r_series, V_0_scalar, a, a_prime, omega_scalar, RHO, dr)

        dT_list.append(dT)
        dM_list.append(dM)

    # Aggregate thrust, torque, power, coefficients
    dT = pd.concat(dT_list)
    dM = pd.concat(dM_list)
    T, M, P, C_T, C_P = compute_totals_and_coefficients(dT, dM, omega_scalar, dr, RHO, R, pd.Series([V_0_scalar]))

    all_results.append({
        "V_0": V_0_scalar,
        "T": T,
        "M": M,
        "P": P,
        "C_T": C_T,
        "C_P": C_P,
    })

    print(f" V_0 = {V_0_scalar:.1f} m/s | P = {P/1e6:.2f} MW | T = {T/1e3:.2f} kN")

#%% -------------------- Plot Power and Thrust Curves --------------------
results_df = pd.DataFrame(all_results).sort_values("V_0")
plot_power_thrust_curves(results_df["V_0"], results_df["P"], results_df["T"])
plot_lift_drag_vs_span(r_s, alpha_list, Cl_list, Cd_list)
plot_induction_factors_vs_span(r_s, a_list, a_prime_list)

#%% -------------------- Summary Table --------------------
print("\nSummary Table:")
print(results_df[["V_0", "P", "T", "C_P", "C_T"]].round(2).to_string(index=False))

#%% -------------------- Done --------------------
end = time()
print(f"\nExecution time: {end - start:.2f} s")
