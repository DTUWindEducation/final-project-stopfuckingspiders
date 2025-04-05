import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO

def load_blade_geometry(filepath):
    """
    Load blade geometry from AeroDyn15 file:
    Extracts BlSpn, BlTwist, BlChord, and BlAFID.
    """
    df = pd.read_csv(
        filepath,
        sep='\s+',
        skiprows=6,  # skip header, units, and metadata
        header=None,
        names=[
            'BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist',
            'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt'
        ]
    )

    # Select and cast relevant columns
    blade_geom = df[['BlSpn', 'BlTwist', 'BlChord', 'BlAFID']]
    blade_geom.loc[:, 'BlAFID'] = blade_geom['BlAFID'].astype(int)

    return blade_geom

def load_operational_strategy(filepath):
    """
    Loads the operational strategy data, skipping the header line.
    """
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=1,  # Skip the malformed "header" row
        header=None,
        names=['WindSpeed', 'PitchAngle', 'RotSpeed', 'AeroPower', 'AeroThrust']
    )
    return df

def load_airfoil_shape(filepath):
    """
    Load normalized airfoil shape coordinates from the airfoil shape file.
    """
    data = np.loadtxt(filepath, skiprows=8)
    return pd.DataFrame(data, columns=["x/c", "y/c"])

def load_all_shapes(shape_dir_paths):
    df_list = []
    for shape_dir_path in shape_dir_paths:
        try:
            df = load_airfoil_shape(shape_dir_path)
            df_list.append(df)
        except Exception as e:
            print(f'[Warning] Failed to load {shape_dir_path}:\n {e}')
    return df_list

import pandas as pd

def load_airfoil_polars(filepaths, skiprows=55):
    """
    Load aerodynamic polar data from an AirfoilInfo file.
    Handles files with variable number of columns.
    """
    df_list = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(
                filepath,
                sep=r'\s+',
                skiprows=skiprows,
                header=None,
                engine='python',
                on_bad_lines='skip'  # skip malformed rows
            )

            if df.shape[1] < 3:
                raise ValueError(f"[Error] Not enough columns in {filepath}.")

            df = df.iloc[:, :3]
            df.columns = ['Alpha', 'Cl', 'Cd']
            df_list.append(df)

        except Exception as e:
            print(f"[Warning] Failed to load {filepath}: {e}")
            return None
    return df_list

def load_all_polars(folder):
    polar_dict = {}
    for i in range(50):
        airfoil_id = i + 1
        filename = f"IEA-15-240-RWT_AeroDyn15_Polar_{i:02d}.dat"
        filepath = os.path.join(folder, filename)

        # Use different skiprows if needed (based on inspection)
        skiprows = 21 if i < 5 else 55
        df = load_airfoil_polars(filepath, skiprows=skiprows)
        if df is not None:
            polar_dict[airfoil_id] = df

    return polar_dict