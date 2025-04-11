from time import time
start = time()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# from scipy.integrate import simps

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

# For development
dfs_geometry = dfs_geometry[0]
df_airfoil_aero_coef = dfs_airfoil_aero_coef[0]

#%% Define boundary conditions:
# Delta r: dr [m]
dr = DR # DR is defined in the config
# Radius: r [m], list r_s [m]
# list from 0 to R with dr increments
r_s = np.arange(0, R + dr, dr) #TODO should r start at 0? Will we get plausible results if r=0?

# For development
r = 1 #TODO replace 'r' with 'r_s'

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

#%% 1. - Initialize induction factors
# Axial induction factors: a [-]
a = 0

# Tangential induction factors: a_prime [-]
a_prime = 0

#%% Create dict with boundary conditions
BC = BoundaryConditions(
    dr=dr,
    r=r,
    Num_Blades=NUMBER_BLADES,
    r_s=r_s,
    V_0=V_0,
    theta_p=theta_p,
    omega=omega,
    BlSpn=BlSpn,
    BlTwist=BlTwist,
    BlChord=BlChord,
    a0=a,
    a0_prime=a_prime
)

#%% Interpolate
# - Blade Twist
# - Chord Length: c [m]
loc_BlTwist, loc_BlChord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)  # TODO use r_s, validate

# Local Solidity: sigma [?]
sigma = calc_local_solidity(BC, loc_BlChord) #TODO use r_s, validate

#%% 2. Compute flow angle 
# Flow Angle: phi [deg]
phi = calc_flow_angle(BC)
phi.name = 'FlowAnlgle (phi)[deg]'

#%% 3. Compute local angle of attack
# Local Angle of Attack: alpha [deg]
alpha = calc_local_angle_of_attack(phi, theta_p, loc_BlTwist) #TODO validate result
alpha.name = 'LocAngleAttack (alpha)[deg]'

#%% 4. Compute local lift and drag force
# Drag Force: C_d [-]
# Lift Force: C_l [-]
C_d, C_l = calc_local_lift_drag_force(alpha, df_airfoil_aero_coef) #TODO validate results

#%% 5. Compute normal and tangential constants
# TODO need to create a function for those values
# TODO Check if phi must be in [deg] or [rad] # From Hannah: DEGREES
C_n, C_t = calc_normal_tangential_constants(phi, C_d, C_l)

#%% 6. Update induction factors
#TODO create function
#TODO validate results
a, a_prime = update_induction_factors(phi, sigma, C_n, C_t)

#%% 7. Check for tolerance
# TODO add go back condition, use global var:
TOLERANCE, BC.a0, BC.a0_prime

#%% 8. Compute local contribution
# loop over all blade elements
# TODO add loop
r_series = pd.Series(data=r_s, index=r_s)

# Local Thrust Contribution: loc_dT [kg m/s^2] = [N]
dT = compute_local_thrust(r_series, V_0, a, RHO, dr) #TODO Validate results

# Local Torque Contribution: loc_dM [kg m/s^2 m] = [Nm]
dM = compute_local_torque(r_series, V_0, a, a_prime, omega, RHO, dr) #TODO Validate results

#%% Integrate (sum) to get total thrust and torque, and compute coefficients
#TODO use integration with trapezoidal rule
T, M, P, C_T, C_P = compute_totals_and_coefficients(dT, dM, omega, dr, RHO, R, V_0)

# Print results
print_summary(T, M, P, C_T, C_P)

#%% END of script
end = time()
print(f"Execution time: {end - start:.4f} seconds")
