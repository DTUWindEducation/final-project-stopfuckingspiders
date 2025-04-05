import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#For Functions
import re
from typing import Union
from pathlib import Path

# For configuration
from config import *
#%% List of Global Variables:
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


# Load module
from src import *
# I (dorian) used an folder with the init etc.
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.plot import *


#%% Functions
def main():
    pass

#%% Read Data
# Blade geometry (chord, twist, airfoil data along the span)

# Operational conditions (wind speed, rotor speed, pitch angle)


#%% Test for main
if __name__ == "__main__":
    pass


#%% My (dorian) code
# ---------- LOAD DATA ----------
blade_df = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
df_strategy = load_operational_strategy(OPERATIONAL_CONDITIONS_FILE_PATH)


airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])
airfoil_shape_paths = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
airfoil_info_paths = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]

dfs_shape = load_all_shapes(airfoil_shape_paths)
#all_polars = load_all_polars(airfoil_info_paths)

# plot airfoil shapes
#plot_airfoil_shapes(shape_data)




#%%Stuff I copied:

# # ---------- BASE PATH ----------
# BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # directory where main.py is located
# INPUT_DIR = os.path.join(BASE_PATH, "inputs", "IEA-15-240-RWT")

# # ---------- FILE PATHS ----------
# BLADE_GEOM_PATH = os.path.join(INPUT_DIR, "IEA-15-240-RWT_AeroDyn15_blade.dat")
# OPS_STRATEGY_PATH = os.path.join(INPUT_DIR, "IEA_15MW_RWT_Onshore.opt")
# POLAR_FOLDER = os.path.join(INPUT_DIR, "Airfoils")
# SHAPE_FOLDER = os.path.join(INPUT_DIR, "Airfoils")
# ROTOR_RADIUS = 120  # meters
# RHO = 1.225  # kg/m^3 (air density)

# # ---------- LOAD DATA ----------
# blade_df = load_blade_geometry(BLADE_GEOM_PATH)
# shape_data = load_all_shapes(SHAPE_FOLDER)
# strategy_df = load_operational_strategy(OPS_STRATEGY_PATH)
# all_polars = load_all_polars(POLAR_FOLDER)

# # plot airfoil shapes
# plot_airfoil_shapes(shape_data)
