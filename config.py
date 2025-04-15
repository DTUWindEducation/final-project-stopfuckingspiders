# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
#from setup.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv()

# Secrets and env vars
#GIT_TOKEN_NAME = os.getenv("GIT_TOKEN_NAME")
#GIT_TOKEN = os.getenv("GIT_TOKEN")

# Load config data
with open(BASE_DIR / 'config.yaml', 'r') as file:
    setup = yaml.safe_load(file)

DATA_PATH = Path(BASE_DIR / setup["DATA_FOLDER"])
REFERENCE_WIND_TURBINE_PATH = Path(DATA_PATH / setup["REFERENCE_WIND_TURBINE"])
BLADE_DEFINITION_INPUT_FILE_PATH= Path(REFERENCE_WIND_TURBINE_PATH / setup["BLADE_DEFINITION_INPUT_FILE"])
OPERATIONAL_CONDITIONS_FILE_PATH = Path(REFERENCE_WIND_TURBINE_PATH / setup["OPERATIONAL_CONDITIONS_FILE"])

AIRFOIL_DATA = Path(REFERENCE_WIND_TURBINE_PATH / setup["AIRFOIL_DATA_FOLDER"])

RESULTS_PATH = Path(BASE_DIR / setup["RESULTS_FOLDER"])

AIRFOIL_SHAPE_IDENTIFIER = setup["AIRFOIL_SHAPE_IDENTIFIER"]
AIRFOIL_INFO_IDENTIFIER = setup["AIRFOIL_INFO_IDENTIFIER"]

#Define global constant parameters
#TODO add vars in to 'config.yaml'
R = setup['RADIUS']  # Rotor Radius [m]
RHO = setup['AIR_DENSITY']  # [kg/m^3] (air density)
NUMBER_BLADES= setup['NUMBER_BLADES'] # [-] Number of blades
TOLERANCE = float(setup["TOLERANCE"])
MAX_ITER = setup['MAX_ITER']
RATED_POWER = setup['RATED_POWER']
CORR = setup['CORRECTION_FACTOR']
EPSILON = setup['EPSILON']
# DR = setup['DR'] # Radial element width
