import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO
from typing import Union
from pathlib import Path

def load_blade_geometry(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load blade nodal geometry from an AeroDyn15 blade input file.

    Parameters
    ----------
    filepath : str or Path
        Path to the AeroDyn15 blade geometry file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cleaned and parsed geometry data,
        including spanwise location (BlSpn), twist (BlTwist),
        chord length (BlChord), and airfoil ID (BlAFID).
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

def load_operational_settings(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the operational conditions for the wind turbine (e.g., wind speed, pitch, RPM).

    Parameters
    ----------
    filepath : str or Path
        Path to the operational conditions file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: WindSpeed, PitchAngle, RotSpeed, AeroPower, AeroThrust.
    """
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=1,
        header=None,
        names=['WindSpeed', 'PitchAngle', 'RotSpeed', 'AeroPower', 'AeroThrust']
    )
    return df

def load_geometry(filepaths: list[Union[str, Path]]) -> list[pd.DataFrame]:
    """
    Load normalized airfoil shape coordinates (x/c, y/c) from coordinate files.

    Parameters
    ----------
    filepaths : list of str or Path
        List of paths to airfoil shape files.

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame contains columns ['x/c', 'y/c'].
    """
    dfs = []
    for filepath in filepaths: 
        data = np.loadtxt(filepath, skiprows=8)
        dfs.append(pd.DataFrame(data, columns=["x/c", "y/c"]))
    return dfs

def load_airfoil_coefficients(filepaths: list[Union[str, Path]]) -> tuple[list[dict], list[pd.DataFrame]]:
    """
    Load aerodynamic polar data (Cl, Cd, Cm) for each airfoil from polar files.

    Parses metadata and polar tables from each file. Also handles unsteady 
    aerodynamic data if included.

    Parameters
    ----------
    filepaths : list of str or Path
        List of paths to airfoil polar files.

    Returns
    -------
    tuple
        headers : list of dict
            Header metadata extracted from each file (if available).
        dfs : list of pd.DataFrame
            Polar data tables with columns: 'Alpha (deg)', 'Cl', 'Cd', 'Cm'.
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

        # Optional sanity checks on Cl could go here

        headers.append(header_data)
        dfs.append(df)

    return headers, dfs

def generate_wind_speed_range(min_speed, max_speed, n_points=50, rated_region=None):
    """
    Generates wind speed range with higher resolution in specified regions
    
    Args:
        min_speed: Minimum wind speed (m/s)
        max_speed: Maximum wind speed (m/s)
        n_points: Total number of points
        rated_region: Tuple of (start, end) for rated wind region
                     Defaults to (10.0, 11.5) if None
    """
    if rated_region is None:
        rated_region = (10.0, 11.5)
    
    rated_points = int(n_points * 0.4)
    other_points = n_points - rated_points
    
    rated_array = np.linspace(rated_region[0], rated_region[1], rated_points)
    other_array = np.concatenate([
        np.linspace(min_speed, rated_region[0]-0.1, other_points//2),
        np.linspace(rated_region[1]+0.1, max_speed, other_points//2)
    ])
    
    return np.sort(np.concatenate([rated_array, other_array]))