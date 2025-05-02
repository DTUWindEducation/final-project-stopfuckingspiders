from WindTurbineModeling.read import *

import pytest

def test_read_blade_definition():
    file_path = r'inputs\IEA-15-240-RWT\IEA-15-240-RWT_AeroDyn15_blade.dat'
    df = read_blade_definition(file_path)

    # Check shape
    assert df.shape[1] == 10  # 10 columns
    assert df.shape[0] == 50  # rows
    # Check column names (only the once used)
    assert "BlSpn" in df.columns
    assert "BlTwist" in df.columns
    assert "BlChord" in df.columns

    #Check values (in row 5)
    # set tolerance to 0.01%
    tolerance = 1e-4
    assert df["BlSpn"].iloc[4] == pytest.approx(9.55101e+00, rel=tolerance) 
    assert df["BlTwist"].iloc[4] == pytest.approx(1.42584e+01, rel=tolerance) 
    assert df["BlChord"].iloc[4] == pytest.approx(5.36733e+00, rel=tolerance) 