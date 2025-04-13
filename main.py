from time import time
start = time()

#%% ------------------------ Imports ------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Helper & typing
import re
from typing import Union
from pathlib import Path
from dataclasses import dataclass

#%% ------------------------ Global Configuration ------------------------
from config import *
# These are defined in config.py:
# - DATA_PATH
# - REFERENCE_WIND_TURBINE_PATH
# - BLADE_DEFINITION_INPUT_FILE_PATH
# - OPERATIONAL_CONDITIONS_FILE_PATH
# - AIRFOIL_DATA
# - RESULTS_PATH
# - AIRFOIL_SHAPE_IDENTIFIER
# - AIRFOIL_INFO_IDENTIFIER
# - R, RHO, NUMBER_BLADES, TOLERANCE, DR

#%% ------------------------ Class Definitions ------------------------
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

#%% ------------------------ Module Imports ------------------------
from WindTurbineModeling import *
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.plot import *
from WindTurbineModeling.equations import *

#%% ------------------------ Load Input Data ------------------------

# Blade geometry
df_blade_input_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)

# Operational settings
df_settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)

# Apply IEA-15MW turbine cut-in / cut-out limits
CUT_IN_WIND_SPEED = 3.0        # [m/s]
CUT_OUT_WIND_SPEED = 25.0      # [m/s]
RATED_WIND_SPEED = 10.6        # [m/s]
RATED_POWER = 15_000_000       # [W]

df_settings = df_settings[
    (df_settings["WindSpeed"] >= CUT_IN_WIND_SPEED) &
    (df_settings["WindSpeed"] <= CUT_OUT_WIND_SPEED)
]

if df_settings.empty:
    print("No valid wind speeds found within cut-in/cut-out limits. Aborting.")
    exit()

V0_range = df_settings["WindSpeed"].min(), df_settings["WindSpeed"].max()
print(f"Loaded {len(df_settings)} valid operating points (Wind speed: {V0_range[0]:.1f}–{V0_range[1]:.1f} m/s)")

#%% ------------------------ Load Airfoil Data ------------------------

# Collect airfoil files
airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])

# Separate geometry and aerodynamic coefficient files
airfoil_shape_paths = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
airfoil_info_paths = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]

# Load files into memory
dfs_geometry = load_geometry(airfoil_shape_paths)
_, dfs_airfoil_aero_coef = load_airfoil_coefficients(airfoil_info_paths)

# Confirm loading
for f_shape, f_info in zip(airfoil_shape_paths, airfoil_info_paths):
    print(f'Shape: {f_shape.stem[:]} => Info: {f_info.stem[:]}')
    # TODO: Run aerodynamic calculations for each pair

# Use the first airfoil dataset for now (development only)
dfs_geometry = dfs_geometry[0]
df_airfoil_aero_coef = dfs_airfoil_aero_coef[0]

#%% ------------------------ Define Boundary Conditions ------------------------

dr = DR  # radial resolution [m]
r_s = np.arange(dr, R + dr, dr)  # span from blade root to tip

V_0 = df_settings['WindSpeed']
theta_p = df_settings['PitchAngle']
omega = df_settings['RotSpeed'] * (2 * np.pi / 60)  # convert to [rad/s]

BlSpn = df_blade_input_data['BlSpn']
BlTwist = df_blade_input_data['BlTwist']
BlChord = df_blade_input_data['BlChord']

# Initialize shared conditions object
BC = BoundaryConditions(
    dr=dr,
    r=0,  # overwritten in loop
    Num_Blades=NUMBER_BLADES,
    r_s=r_s,
    V_0=V_0,
    theta_p=theta_p,
    omega=omega,
    BlSpn=BlSpn,
    BlTwist=BlTwist,
    BlChord=BlChord,
    a0=0.0,
    a0_prime=0.0
)

#%% ------------------------ Iteration Loop ------------------------

# Initialize result containers
dT_list = []
dM_list = []
a_list = []
a_prime_list = []

V_0_scalar = V_0.mean()
omega_scalar = omega.mean()

TOLERANCE = 1e-8
MAX_ITER = 100

# Iterate over all radial sections
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
        alpha = calc_local_angle_of_attack(phi, theta_p, loc_BlTwist)
        C_d, C_l = calc_local_lift_drag_force(alpha, df_airfoil_aero_coef)
        C_n, C_t = calc_normal_tangential_constants(phi, C_d, C_l)

        # Apply Prandtl's tip loss correction
        F = calc_prandtl_tip_loss(NUMBER_BLADES, R, r, phi)

        # Update axial and tangential induction with Glauert correction
        a_new, a_prime_new = update_induction_factors(phi, sigma, C_n, C_t, F, a)

        a_new = float(np.mean(a_new))
        a_prime_new = float(np.mean(a_prime_new))

        # Convergence check
        if abs(a_new - a) < TOLERANCE and abs(a_prime_new - a_prime) < TOLERANCE:
            break

        a, a_prime = a_new, a_prime_new

    # Store converged results
    a_list.append(a)
    a_prime_list.append(a_prime)

    r_series = pd.Series([r], index=[r])
    dT = compute_local_thrust(r_series, V_0_scalar, a, RHO, dr)
    dM = compute_local_torque(r_series, V_0_scalar, a, a_prime, omega_scalar, RHO, dr)

    print(f"r = {r:.2f} m | a = {a:.4f}, a' = {a_prime:.4f}")
    print(f"  alpha = {alpha.mean():.2f} deg | Cl = {C_l.mean():.2f}, Cd = {C_d.mean():.2f}")
    print(f"  Cn = {C_n.mean():.2f}, Ct = {C_t.mean():.2f}")
    print(f"  dT = {dT.values[0]:.2e} N | dM = {dM.values[0]:.2e} Nm\n")

    dT_list.append(dT)
    dM_list.append(dM)

#%% ------------------------ Final Calculations ------------------------

dT = pd.concat(dT_list, axis=0).sort_index()
dM = pd.concat(dM_list, axis=0).sort_index()
a_series = pd.Series(a_list, index=r_s, name="a")
a_prime_series = pd.Series(a_prime_list, index=r_s, name="a_prime")

# Final rotor integration and performance computation
# TODO: Optionally plot spanwise distributions of a, alpha, dT, etc.
T, M, P, C_T, C_P = compute_totals_and_coefficients(dT, dM, omega_scalar, dr, RHO, R, V_0)

if P > RATED_POWER:
    print(f"Warning: Simulated power output {P / 1e6:.2f} MW exceeds rated capacity (15.0 MW)")

print_summary(T, M, P, C_T, C_P)

#%% ------------------------ End Timer ------------------------
end = time()
print(f"Execution time: {end - start:.4f} seconds")
