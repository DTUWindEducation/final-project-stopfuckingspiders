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

DATA_PATH = BASE_DIR / setup["DATA_FOLDER"]
REFERENCE_WIND_TURBINE = DATA_PATH / setup["REFERENCE_WIND_TURBINE"]
RESULTS_PATH = BASE_DIR / setup["RESULTS_FOLDER"]
AIRFOIL_DATA_FOLDER = REFERENCE_WIND_TURBINE / setup["AIRFOIL_DATA_FOLDER"]
AIRFOIL_DATA = REFERENCE_WIND_TURBINE / setup["AIRFOIL_DATA"]

