from time import time
start = time()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from scipy.integrate import simps

#For Functions
import re
from typing import Union
from pathlib import Path

#%% ---------- Global Variables ----------
# List of Global Variables defined in the config file:
from config import *
# DATA_PATH
# REFERENCE_WIND_TURBINE_PATH
# BLADE_DEFINITION_INPUT_FILE_PATH # input file for blade geometry
# OPERATIONAL_CONDITIONS_FILE_PATH
# AIRFOIL_DATA
# RESULTS_PATH
# AIRFOIL_SHAPE_IDENTIFIER
# AIRFOIL_INFO_IDENTIFIER
# R
# RHO
# B

from dataclasses import dataclass
@dataclass
class BoundaryConditions:
    dr: float
    r: float
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
#%% -- FUNCTIONS --

# - Main -
def main():
    pass

#%% -- IF MAIN --
if __name__ == "__main__":
    pass

#%% -- LOAD DATA --
# -- Blade Data Input --
dfs_blade_input_data = load_blade_geometry([BLADE_DEFINITION_INPUT_FILE_PATH])
# Since we only have one input file.
df_blade_input_data = dfs_blade_input_data[0] 

# -- Operational Settings --
df_settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)
#TODO Can be adjusted so that multiple setting can be loaded

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
unsteady_aerodynamics_coefs, dfs_airfoil_aero_coef = load_airfoil_coefficients(airfoil_info_paths)

df_airfoil_aero_coef = dfs_airfoil_aero_coef[0]

#%% Define boundary conditions:
# Delta r: dr [m]
dr = 0.1
# Radius: r [m], list r_s [m]
# list from 0 to R with dr increments
r_s = np.arange(0, R+dr, dr)
r = 1
# Inflow wind speed: V_0 [m/s]
V_0 = df_settings['WindSpeed']
V_0.name = 'WindSpeed (V_0)[m/s]'

# Blade pitch angle: theta_p [deg]
theta_p = df_settings['PitchAngle']
theta_p.name = 'BladePitchAngle (theta_p)[deg]'

#Rotational speed: omega [rpm] -> [1/s]
omega = df_settings['RotSpeed']/60
omega.name = 'RotSpeed (omega)[1/s]'

BlSpn = df_blade_input_data['BlSpn']
BlTwist = df_blade_input_data['BlTwist']
BlChord = df_blade_input_data['BlChord']


#%% 1. - Initialize induction factors
# Axial induction factors: a [?]
a = 0

# Tangential induction factors: a_prime [?]
a_prime = 0

#%% Create dict with boudary conditions

BC = BoundaryConditions(
    dr=dr,
    r=r,
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
loc_BlTwist = np.interp(r, BlSpn, BlTwist) #[deg]
# - Chord Length: c [m]
c = loc_BlTwist = np.interp(r, BlSpn, BlChord) #[deg]

# Local Solidity: sigma [?]
sigma = calc_local_solidity(r, c, B)

#%% 2. Compute flow angle 
# Flow Angle: phi [deg]
# Flow anlge phi_xr is a 2D-Array with results of phi(r) for each r

phi_xr = get_flow_angle(BC)
r = 1
phi = phi_xr[:,0].values
#%% 3. Compute local angle of attack
# Local Angle of Attack: alpha [deg]

# 3.b calc local angle of attack
alpha = calc_local_angle_of_attack(phi, theta_p, loc_BlTwist)
alpha.name = 'LocAngleAttack (alpha)[deg]'

#%% 4. Compute local lift and drag force
# Drag Force: C_d [-]
# Lift Force: C_l [-]

C_d, C_l = calc_local_lift_drag_force(alpha, df_airfoil_aero_coef)

#%% 5.Compute 
# TODO what is C_n and C_t
# TODO need to create a function for those values
# TODO Check if phi must be in [deg] or [red]

# ???: C_n [-]
C_n = C_l * np.cos(phi) + C_d * np.sin(phi)
C_n.name = 'Name (C_n)[deg]'

# ???: C_t [-]
C_t = C_l * np.sin(phi) + C_d * np.cos(phi)
C_t.name = 'Name (C_t)[deg]'

#%% 6. Update induction factors
# Axial induction factors: a [?]
denominator = (4 * np.sin(phi)**2) / ((sigma*C_n)+1)
a = 1/(denominator)
a.name = 'Axial induction factors (a)[-]'

# Tangential induction factors: a_prime [?]
denominator = (4 * np.sin(phi)* np.cos(phi)) / ((sigma*C_t)-1)
a_prime = 1/(denominator)
a_prime.name = 'Tangential induction factors (a_prime)[-]'

#%% 7. Check for tolerance
# TODO check to tolerance and build in "go back..." 

#%% 8. Compute local contribution
# loop over all blade elements
# TODO add loop

# Local Thrust Contribution: loc_dT [?]
# TODO please validate
dT = 4*np.pi * r * RHO * V_0**2 * a * (1-a) * dr 

# Local Torque Contribution: loc_dM [?]
# TODO please validate
#dM = 4*np.pi * r**3 * RHO * V_0**2 * omega * a_prime (1-a_prime) * dr

#%% END of script

end = time()
print(f"Execution time: {end - start:.4f} seconds")