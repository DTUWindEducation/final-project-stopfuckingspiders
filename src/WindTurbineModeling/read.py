from typing import Union
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pandas as pd


def read_blade_definition(file_path: Union[str, Path]):
    """
    Reads a blade definition file and parses its content into a pandas
    DataFrame. The function expects the file to contain numeric data lines
    corresponding to blade properties, with each line containing at least 10
    numeric values. The resulting DataFrame will have predefined column names.
    Parameters:
    -----------
    file_path : Union[str, Path]
        The path to the blade definition file to be read.
    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the parsed blade data with
        the following columns:
        - "BlSpn": Blade span
        - "BlCrvAC": Blade curve along the chord axis
        - "BlSwpAC": Blade sweep along the chord axis
        - "BlCrvAng": Blade curve angle
        - "BlTwist": Blade twist angle
        - "BlChord": Blade chord length
        - "BlAFID": Blade airfoil ID
        - "BlCb": Blade camber
        - "BlCenBn": Blade center of bending (normal)
        - "BlCenBt": Blade center of bending (tangential)
    """
    # Does the same as load_blade_geometry
    with open(file_path, "r") as file:
        content = file.readlines()

    # Column names from the header
    column_names = [
        "BlSpn", "BlCrvAC", "BlSwpAC", "BlCrvAng", "BlTwist", "BlChord",
        "BlAFID", "BlCb", "BlCenBn", "BlCenBt"
    ]

    # Extract only the lines with numeric data
    data_lines = [
        line for line in content
        if re.match(r"^\s*[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+", line)
    ]

    # Convert the lines into a DataFrame
    data = [list(map(float, line.split()[:10])) for line in data_lines]
    df = pd.DataFrame(data, columns=column_names)

    return df

def get_files_by_extension(folder_path: Union[str, Path],
                           extensions: list[str] = [".csv"]) -> list[Path]:
    """
    Scans a folder (including sub-folders) and returns a list of file paths
    that match any of the specified extensions.

    Parameters:
        folder_path (Union[str, Path]): Path to the folder to scan.
        extensions (List[str], optional): List of file extensions to look for.
                                          Each extension should include the
                                          leading dot (e.g., '.csv').
                                          Defaults to ['.csv'].

    Returns:
        List[Path]: List of matching file paths as Path objects.
    """
    folder_path = Path(folder_path)
    result = []

    for ext in extensions:
        # Ensure the extension starts with a dot
        if not ext.startswith("."):
            ext = f".{ext}"
        result.extend(folder_path.rglob(f"*{ext}"))

    return result

