import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file (if it exists)
# This allows sensitive or configurable settings to be stored in a .env file and accessed via environment variables.
load_dotenv()

# Load configuration data from config.yaml in the root folder
# The config.yaml file contains project-specific settings such as folder paths and constants.
with open(PROJECT_ROOT / 'config.yaml', 'r') as file:
    setup = yaml.safe_load(file)

# Set paths relative to the project root
# These paths are derived from the configuration file and are used throughout the project.
DATA_PATH = PROJECT_ROOT / setup["DATA_FOLDER"]  # Path to the data folder
REFERENCE_WIND_TURBINE_PATH = DATA_PATH / setup["REFERENCE_WIND_TURBINE"]  # Path to the reference wind turbine folder
BLADE_DEFINITION_INPUT_FILE_PATH = REFERENCE_WIND_TURBINE_PATH / setup["BLADE_DEFINITION_INPUT_FILE"]  # Path to the blade definition input file
OPERATIONAL_CONDITIONS_FILE_PATH = REFERENCE_WIND_TURBINE_PATH / setup["OPERATIONAL_CONDITIONS_FILE"]  # Path to the operational conditions file
AIRFOIL_DATA = REFERENCE_WIND_TURBINE_PATH / setup["AIRFOIL_DATA_FOLDER"]  # Path to the airfoil data folder
RESULTS_PATH = PROJECT_ROOT / setup["RESULTS_FOLDER"]  # Path to the results folder

# Airfoil shape identifiers
# These identifiers are used to parse or identify specific airfoil-related data in files.
AIRFOIL_SHAPE_IDENTIFIER = setup["AIRFOIL_SHAPE_IDENTIFIER"]  # Identifier for airfoil shape data
AIRFOIL_INFO_IDENTIFIER = setup["AIRFOIL_INFO_IDENTIFIER"]  # Identifier for airfoil information data

# Define global constant parameters from the config.yaml
# These constants are used in the wind turbine modeling calculations.
R = setup['RADIUS']  # Rotor Radius [m]
RHO = setup['AIR_DENSITY']  # Air density [kg/m^3]
NUMBER_BLADES = setup['NUMBER_BLADES']  # Number of blades on the wind turbine
TOLERANCE = float(setup["TOLERANCE"])  # Tolerance for numerical calculations
MAX_ITER = setup['MAX_ITER']  # Maximum number of iterations for numerical methods
RATED_POWER = setup['RATED_POWER']  # Rated power of the wind turbine
CORR = setup['CORRECTION_FACTOR']  # Correction factor for specific calculations
EPSILON = float(setup['EPSILON'])  # Small numerical value used to avoid division by zero or other numerical issues
