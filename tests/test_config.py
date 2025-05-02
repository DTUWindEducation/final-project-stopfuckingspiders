
from pathlib import Path
# tests/test_config.py
import pytest
import yaml
from pathlib import Path

import WindTurbineModeling.config as config

def test_gitanswers_exists():
    """Verify that the file 'config.yaml' exists"""
    
    filename = 'config.yaml'  # the file must have this name
    
    p = Path(filename)  # create a pathlib.Path object, which has useful methods
    is_file = p.is_file()  # get True or False depending on if file exists
    
    assert is_file  # throw an error if the file doesn't exist
@pytest.fixture(scope="module")
def config_data():
    config_path = config.PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_project_root_is_directory():
    assert config.PROJECT_ROOT.is_dir()


def test_config_file_exists():
    assert (config.PROJECT_ROOT / 'config.yaml').exists()


def test_data_path(config_data):
    expected = config.PROJECT_ROOT / config_data["DATA_FOLDER"]
    assert config.DATA_PATH == expected


def test_reference_wind_turbine_path(config_data):
    expected = config.DATA_PATH / config_data["REFERENCE_WIND_TURBINE"]
    assert config.REFERENCE_WIND_TURBINE_PATH == expected


def test_blade_definition_input_path(config_data):
    expected = config.REFERENCE_WIND_TURBINE_PATH / config_data["BLADE_DEFINITION_INPUT_FILE"]
    assert config.BLADE_DEFINITION_INPUT_FILE_PATH == expected


def test_operational_conditions_path(config_data):
    expected = config.REFERENCE_WIND_TURBINE_PATH / config_data["OPERATIONAL_CONDITIONS_FILE"]
    assert config.OPERATIONAL_CONDITIONS_FILE_PATH == expected


def test_airfoil_data_path(config_data):
    expected = config.REFERENCE_WIND_TURBINE_PATH / config_data["AIRFOIL_DATA_FOLDER"]
    assert config.AIRFOIL_DATA == expected


def test_results_path(config_data):
    expected = config.PROJECT_ROOT / config_data["RESULTS_FOLDER"]
    assert config.RESULTS_PATH == expected


def test_constants(config_data):
    assert isinstance(config.R, (int, float))
    assert config.R == config_data["RADIUS"]

    assert isinstance(config.RHO, (int, float))
    assert config.RHO == config_data["AIR_DENSITY"]

    assert isinstance(config.NUMBER_BLADES, int)
    assert config.NUMBER_BLADES == config_data["NUMBER_BLADES"]

    assert isinstance(config.TOLERANCE, float)
    assert config.TOLERANCE == float(config_data["TOLERANCE"])

    assert isinstance(config.MAX_ITER, int)
    assert config.MAX_ITER == config_data["MAX_ITER"]

    assert isinstance(config.RATED_POWER, (int, float))
    assert config.RATED_POWER == config_data["RATED_POWER"]

    assert isinstance(config.CORR, (int, float))
    assert config.CORR == config_data["CORRECTION_FACTOR"]

    assert isinstance(config.EPSILON, (int, float))
    assert config.EPSILON == float(config_data["EPSILON"])


def test_airfoil_identifiers(config_data):
    assert config.AIRFOIL_SHAPE_IDENTIFIER == config_data["AIRFOIL_SHAPE_IDENTIFIER"]
    assert config.AIRFOIL_INFO_IDENTIFIER == config_data["AIRFOIL_INFO_IDENTIFIER"]