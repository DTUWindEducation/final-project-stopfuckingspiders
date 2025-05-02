# src/bem_analysis/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file (if it exists)
load_dotenv()

# Load config data from config.yaml in the root folder
with open(PROJECT_ROOT / 'config.yaml', 'r') as file:
    setup = yaml.safe_load(file)

# Set paths relative to the project root
DATA_PATH = PROJECT_ROOT / setup["DATA_FOLDER"]
REFERENCE_WIND_TURBINE_PATH = DATA_PATH / setup["REFERENCE_WIND_TURBINE"]
BLADE_DEFINITION_INPUT_FILE_PATH = REFERENCE_WIND_TURBINE_PATH / setup["BLADE_DEFINITION_INPUT_FILE"]
OPERATIONAL_CONDITIONS_FILE_PATH = REFERENCE_WIND_TURBINE_PATH / setup["OPERATIONAL_CONDITIONS_FILE"]
AIRFOIL_DATA = REFERENCE_WIND_TURBINE_PATH / setup["AIRFOIL_DATA_FOLDER"]
RESULTS_PATH = PROJECT_ROOT / setup["RESULTS_FOLDER"]

# Airfoil shape identifiers
AIRFOIL_SHAPE_IDENTIFIER = setup["AIRFOIL_SHAPE_IDENTIFIER"]
AIRFOIL_INFO_IDENTIFIER = setup["AIRFOIL_INFO_IDENTIFIER"]

# Define global constant parameters from the config.yaml
R = setup['RADIUS']  # Rotor Radius [m]
RHO = setup['AIR_DENSITY']  # [kg/m^3]
NUMBER_BLADES = setup['NUMBER_BLADES']  # [-] Number of blades
TOLERANCE = float(setup["TOLERANCE"])
MAX_ITER = setup['MAX_ITER']
RATED_POWER = setup['RATED_POWER']
CORR = setup['CORRECTION_FACTOR']
EPSILON = float(setup['EPSILON'])
