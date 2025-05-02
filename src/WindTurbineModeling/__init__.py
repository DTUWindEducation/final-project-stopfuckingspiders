import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

#from scipy.integrate import simps

#For Functions
import re
from typing import Union
from pathlib import Path

#%% Load Wind Turbine Model Module
from src.WindTurbineModeling.read import *
from src.WindTurbineModeling.load import *
from src.WindTurbineModeling.plot import *
from src.WindTurbineModeling.equations import *
