# main.py
from time import time
start = time()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# For Functions
import re
from typing import Union
from pathlib import Path

#%% ---------- Global Variables ----------
from config import *
# List of Global Variables defined in the config file:
# -DATA_PATH
# -REFERENCE_WIND_TURBINE_PATH
# -BLADE_DEFINITION_INPUT_FILE_PATH # input file for blade geometry
# -OPERATIONAL_CONDITIONS_FILE_PATH
# -AIRFOIL_DATA
# -RESULTS_PATH
# -AIRFOIL_SHAPE_IDENTIFIER
# -AIRFOIL_INFO_IDENTIFIER
# -R
# -RHO
# -NUMBER_BLADES
# -TOLERANCE
# -DR

#%% Define classes
# Class for Boundary Conditions to pass in functions etc.
from dataclasses import dataclass
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

#%% Load Wind Turbine Model Module
from WindTurbineModeling import *
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.plot import *
from WindTurbineModeling.equations import *
#*******************************************************************************

#%% -- LOAD DATA --
# -- Blade Data Input --
df_blade_input_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)

# -- Operational Settings --
df_settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)

# -- Wind Speed Filtering Based on IEA-15MW Reference Constraints --
CUT_IN_WIND_SPEED = 3.0        # [m/s]
CUT_OUT_WIND_SPEED = 25.0      # [m/s]
RATED_WIND_SPEED = 10.6        # [m/s]
RATED_POWER = 15_000_000       # [W]

# Filter operational settings to valid range
df_settings = df_settings[
    (df_settings["WindSpeed"] >= CUT_IN_WIND_SPEED) &
    (df_settings["WindSpeed"] <= CUT_OUT_WIND_SPEED)
]

if df_settings.empty:
    print("No valid wind speeds found within cut-in/cut-out limits. Aborting.")
    exit()

V0_range = df_settings["WindSpeed"].min(), df_settings["WindSpeed"].max()
print(f"Loaded {len(df_settings)} valid operating points (Wind speed: {V0_range[0]:.1f}–{V0_range[1]:.1f} m/s)")

# -- Aerodynamic Properties --
# Load all files in a list with the aerodynamic properties (geometry and coefficients)
airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])

# -- Airfoil Shape (geometry) --
# Create a list with all files describing the airfoil geometry
airfoil_shape_paths = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
# Get a list of dataframes with the airfoil geometries
dfs_geometry = load_geometry(airfoil_shape_paths)

# -- Aerodynamic Coefficients --
# Create a list with all files describing the airfoil coefficients
airfoil_info_paths = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]
# Get a list of dict and a list of dataframes
# dict: contain header information for the coefficient (only if unsteady aerodynamics data is included)
# dataframe: contains the coefficients
_, dfs_airfoil_aero_coef = load_airfoil_coefficients(airfoil_info_paths)

#%% Start iteration though each airfoil -info and -geometry file
for f_shape, f_info in zip(airfoil_shape_paths, airfoil_info_paths):
    print(f'Shape: {f_shape.stem[:]} => Info: {f_info.stem[:]}')
    #TODO equation should be run here for each airfoil

# For development: use only the first geometry and coefficient set
dfs_geometry = dfs_geometry[0]
df_airfoil_aero_coef = dfs_airfoil_aero_coef[0]

#%% Define boundary conditions:
# Delta r: dr [m]
dr = DR  # DR is defined in the config

# Radius span: list from 0 to R with dr increments
r_s = np.arange(3, R + dr, dr)  # TODO should r start at 0?

# Inflow wind speed: V_0 [m/s]
V_0 = df_settings['WindSpeed']
V_0.name = 'WindSpeed (V_0)[m/s]'

# Blade pitch angle: theta_p [deg]
theta_p = df_settings['PitchAngle']
theta_p.name = 'BladePitchAngle (theta_p)[deg]'

# Rotational speed: omega [rpm] -> [1/s]
omega = df_settings['RotSpeed'] / 60
omega.name = 'RotSpeed (omega)[1/s]'

# Local Blade Span at node [m]
BlSpn = df_blade_input_data['BlSpn']

# Local Blade Twist at node [deg]
BlTwist = df_blade_input_data['BlTwist']

# Local Blade Chord at node [m]
BlChord = df_blade_input_data['BlChord']

#%% Create dict with boundary conditions
BC = BoundaryConditions(
    dr=dr,
    r=0,  # Overwritten in loop
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

#%% Loop over full span with convergence check
dT_list = []
dM_list = []
a_list = []
a_prime_list = []

V_0_scalar = V_0.mean()
omega_scalar = omega.mean()
TOLERANCE = 1e-8
MAX_ITER = 100

for r in r_s:
    loc_BlTwist, loc_BlChord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)
    BC.r = r
    sigma = calc_local_solidity(BC, loc_BlChord)

    # Initial guess for induction factors
    a = 0
    a_prime = 0

    # Iterate to converge
    for _ in range(MAX_ITER):
        BC.a0 = a
        BC.a0_prime = a_prime

        phi = calc_flow_angle(BC)
        alpha = calc_local_angle_of_attack(phi, theta_p, loc_BlTwist)
        C_d, C_l = calc_local_lift_drag_force(alpha, df_airfoil_aero_coef)
        C_n, C_t = calc_normal_tangential_constants(phi, C_d, C_l)
        a_new, a_prime_new = update_induction_factors(phi, sigma, C_n, C_t)

        a_new = float(np.mean(a_new))
        a_prime_new = float(np.mean(a_prime_new))

        if abs(a_new - a) < TOLERANCE and abs(a_prime_new - a_prime) < TOLERANCE:
            break

        a, a_prime = a_new, a_prime_new

    # Store results
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

#%% Assemble into full Series
dT = pd.concat(dT_list, axis=0).sort_index()
dM = pd.concat(dM_list, axis=0).sort_index()
a_series = pd.Series(a_list, index=r_s, name="a")
a_prime_series = pd.Series(a_prime_list, index=r_s, name="a_prime")

#%% Final integration and performance
#TODO: optionally plot distributions of a, alpha, dT, etc.
T = np.trapz(dT, dx=dr)
M = np.trapz(dM, dx=dr)
P = np.trapz((dM / dr).values * omega_scalar, dx=dr)
C_T, C_P = compute_rotor_coefficients(T, P, RHO, R, V_0)

if P > RATED_POWER:
    print(f"Warning: Simulated power output {P / 1e6:.2f} MW exceeds rated capacity (15.0 MW)")

print_summary(T, M, P, C_T, C_P)

#%% END
end = time()
print(f"Execution time: {end - start:.4f} seconds")
