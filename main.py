import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#For Functions
import re
from typing import Union
from pathlib import Path

#%% List of Global Variables defined in the config file:
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

#%% Load Wind Turbine Model Module
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.plot import *


#%% Functions
def main():
    pass

#%% Test for main
if __name__ == "__main__":
    pass

#%% ---------- LOAD DATA ----------
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

# plot airfoil shapes
#plot_airfoil_shapes(shape_data)
