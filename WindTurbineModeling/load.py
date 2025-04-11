import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO
import pandas as pd
from typing import Union
from pathlib import Path


def load_blade_geometry(filepath: Union[str, Path]):
    """
    Load blade nodal geometry from an AeroDyn15 blade input file.
    Inputs from file:
    - NumBlNds: Number of blade nodes (must match number of rows in table).
    - BlSpn: Spanwise location from root [m], must start at 0 and increase.
    - BlCrvAC: Out-of-plane offset of aerodynamic center [m], positive downwind.
    - BlSwpAC: In-plane offset of aerodynamic center [m], positive opposite rotation.
    - BlCrvAng: Angle of normal vector from airfoil plane due to curvature [deg].
    - BlTwist: Local aerodynamic twist angle [deg], positive to feather.
    - BlChord: Local chord length [m].
    - BlAFID: Airfoil ID corresponding to entry in airfoil data table.
    Each row in the file defines one node along the blade, from root to tip.
    """


    df = pd.read_csv(
        filepath,
        sep=r'\s+',
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

def load_operational_settings(filepath: Union[str, Path]):
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

def load_geometry(filepaths: list[Union[str, Path]]):
    """
    Load normalized airfoil shape coordinates from the airfoil shape file.
    """
    dfs = []
    for filepath in filepaths: 
        data = np.loadtxt(filepath, skiprows=8)
        dfs.append(pd.DataFrame(data, columns=["x/c", "y/c"]))
    return dfs

def load_airfoil_coefficients(filepaths: list[Union[str, Path]]):
    """
    Load aerodynamic polar data from a list of airfoil files.

    Each file is expected to contain a table of aerodynamic coefficients
    (angle of attack, lift, drag, moment), preceded by optional headers.
    This function extracts the aerodynamic table from each file and applies
    consistency checks:
      - Skips files with nearly constant Cl values (flat dummy polars)
      - Flips Cl sign if it decreases with increasing alpha

    Parameters:
        filepaths (list of Path or str): List of file paths to airfoil polar data.

    Returns:
        headers (list of dict): Metadata parsed from each file, if available.
        dfs (list of pd.DataFrame): DataFrames containing columns:
            'Alpha (deg)', 'Cl', 'Cd', 'Cm'
    """
    headers = []
    dfs = []

    for filepath in filepaths:
        header_data = {}
        table_data = []

        with open(filepath, "r") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith("!"):
                i += 1
                continue

            parts = line.split("!")
            data_part = parts[0].strip()
            tokens = data_part.split()

            if len(tokens) >= 2:
                value = tokens[0]
                key = tokens[1]
                header_data[key] = value

            if "InclUAdata" in data_part:
                if value.lower() == "true":
                    i += 1
                    while not lines[i].strip().startswith("! Table of aerodynamics coefficients"):
                        ua_line = lines[i].strip()
                        if ua_line and not ua_line.startswith("!"):
                            ua_parts = ua_line.split("!")
                            ua_data = ua_parts[0].strip().split()
                            if len(ua_data) >= 2:
                                ua_value = ua_data[0]
                                ua_key = ua_data[1]
                                header_data[ua_key] = ua_value
                        i += 1
                else:
                    while not lines[i].strip().startswith("! Table of aerodynamics coefficients"):
                        i += 1

            i += 1
            if lines[i].strip().startswith("!    Alpha"):
                i += 2
                break

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("!"):
                i += 1
                continue
            parts = line.split()
            row = [float(p) for p in parts]
            table_data.append(row)
            i += 1

        df = pd.DataFrame(table_data, columns=["Alpha (deg)", "Cl", "Cd", "Cm"])

        # Sanity checks on Cl
        cl_std = df["Cl"].std()
        cl_at_pos_alpha = df[df["Alpha (deg)"] > 10]["Cl"].mean()
        cl_at_neg_alpha = df[df["Alpha (deg)"] < -10]["Cl"].mean()

        if cl_std < 0.01:
            print(f"❌ Skipping '{filepath.name}': Cl appears flat (std={cl_std:.5f})")
            continue

        if cl_at_pos_alpha < cl_at_neg_alpha:
            print(f"⚠️ Flipping Cl in '{filepath.name}': increasing alpha gives decreasing Cl")
            df["Cl"] = -df["Cl"]

        headers.append(header_data)
        dfs.append(df)

    return headers, dfs



