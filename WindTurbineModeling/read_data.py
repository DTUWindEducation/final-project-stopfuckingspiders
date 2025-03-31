import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import config as c

#For Functions
import re
from typing import Union
from pathlib import Path

def read_blade_definition(file_path: Union[str, Path]):

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
