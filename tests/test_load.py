from src.WindTurbineModeling.load import *
import pytest

def test_load_blade_geometry():
    file_path = r"inputs\IEA-15-240-RWT\IEA-15-240-RWT_AeroDyn15_blade.dat"
    df = load_blade_geometry(file_path)

    # Check shape
    assert df.shape[1] == 4  # 4 columns
    assert df.shape[0] == 50  # rows

    # Check column names (only the once used)
    assert "BlSpn" in df.columns
    assert "BlTwist" in df.columns
    assert "BlChord" in df.columns

    # Check values (in row 5)
    row= 5
    # set tolerance to 0.01%
    tolerance = 1e-4
    assert df["BlSpn"].iloc[row-1] == pytest.approx(9.55101e+00, rel=tolerance) 
    assert df["BlTwist"].iloc[row-1] == pytest.approx(1.42584e+01, rel=tolerance) 
    assert df["BlChord"].iloc[row-1] == pytest.approx(5.36733e+00, rel=tolerance) 

def test_load_operational_settings():
    file_path = r"inputs\IEA-15-240-RWT\IEA_15MW_RWT_Onshore.opt"
    df = load_operational_settings(file_path)
    col_names_exp = ['WindSpeed', 'PitchAngle', 'RotSpeed', 'AeroPower', 'AeroThrust']

    # Check shape
    assert df.shape[1] == 5  # 5 columns
    assert df.shape[0] == 17  # 17 rows

    # Check column names
    for col_name in col_names_exp:
        assert col_name in df.columns
    
    # Check values (in row 5)
    row = 5 
    # set tolerance to 0%
    tolerance = 0
    assert df["WindSpeed"].iloc[row-1] == pytest.approx(8, rel=tolerance) 
    assert df["PitchAngle"].iloc[row-1] == pytest.approx(5.35e-4, rel=tolerance) 
    assert df["RotSpeed"].iloc[row-1] == pytest.approx(5.6819, rel=tolerance) 
    assert df["AeroPower"].iloc[row-1] == pytest.approx(6852.108460, rel=tolerance) 
    assert df["AeroThrust"].iloc[row-1] == pytest.approx(1295.039597, rel=tolerance) 


def test_load_geometry():
    file_path = [r"inputs\IEA-15-240-RWT\Airfoils\IEA-15-240-RWT_AF48_Coords.txt"]
    df = load_geometry(file_path)[0]

    # Check shape
    assert df.shape[1] == 2  # 5 columns
    assert df.shape[0] == 200  # 200 rows

    # Check column names
    col_names_exp = ["x/c", "y/c"]
    for col_name in col_names_exp:
        assert col_name in df.columns

    # Check values (in row 5)
    row = 5
    # set tolerance to 0.01%
    tolerance = 1e-4
    exp_val = 9.04146e-01
    assert df["x/c"].iloc[row-1] == pytest.approx(exp_val, rel=tolerance) 
    exp_val = 6.03552e-03
    assert df["y/c"].iloc[row-1] == pytest.approx(exp_val, rel=tolerance) 
    

def test_load_airfoil_coefficients():
    file_paths = [r"inputs\IEA-15-240-RWT\Airfoils\IEA-15-240-RWT_AeroDyn15_Polar_03.dat", 
                    r"inputs\IEA-15-240-RWT\Airfoils\IEA-15-240-RWT_AeroDyn15_Polar_45.dat"]
    
    _, dfs = load_airfoil_coefficients(file_paths)
    
    # Values of row 5 in: IEA-15-240-RWT_AeroDyn15_Polar_03.dat
    vals_Polar_03 = [-1.68e+02,2.31621e-01,2.83762e-01,5.64573e-02]
    
    # Values of row 5 in: IEA-15-240-RWT_AeroDyn15_Polar_45.dat
    vals_Polar_45 = [-1.68e+02,2.83635e-01,6.23536e-02,3.88683e-01]
    
    vals = [vals_Polar_03, vals_Polar_45]

    for df, val in zip(dfs, vals):
        # Check shape
        assert df.shape[1] == 4  # 5 columns
        assert df.shape[0] == 200  # 200 rows

        # Check column names
        col_names_exp = ["Alpha (deg)", "Cl", "Cd", "Cm"]
        for col_name in col_names_exp:
            assert col_name in df.columns

        # Check values (in row 5)
        row = 5
        # set tolerance to 0.01%
        tolerance = 1e-4
        for col_name, v in  zip(col_names_exp, val):
               assert df[col_name].iloc[row-1] == pytest.approx(v, rel=tolerance) 
 