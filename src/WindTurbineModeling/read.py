import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas as pd
from pathlib import Path
#import config as c

#For Functions
import re
from typing import Union
from pathlib import Path

def read_blade_definition(file_path: Union[str, Path]):
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

def get_files_by_extension(folder_path: Union[str, Path], extensions: list[str] = [".csv"]) -> list[Path]:
    """
    Scans a folder (including sub-folders) and returns a list of file paths
    that match any of the specified extensions.

    Parameters:
        folder_path (Union[str, Path]): Path to the folder to scan.
        extensions (List[str], optional): List of file extensions to look for.
                                          Each extension should include the leading
                                          dot (e.g., '.csv'). Defaults to ['.csv'].

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

