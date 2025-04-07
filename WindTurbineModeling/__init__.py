import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from scipy.integrate import simps

#For Functions
import re
from typing import Union
from pathlib import Path

#%% Load Wind Turbine Model Module
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.plot import *
from WindTurbineModeling.equations import *


def get_flow_angle(BC):
      
    phi_list = [] 
    
    for r in BC.r_s:
        phi_r = calc_flow_angle(BC.a0, BC.a0_prime, BC.V_0, BC.omega, BC.r)
        phi_list.append(np.array(phi_r))  # convert phi_r to np array
    
    # Stack into 2D array: shape (n_r, n_phi)
    phi_array = np.stack(phi_list)  # shape: (len(r_s), len(V_0))
    del phi_list
    # Build xarray: transpose to (phi, r)
    phi_xr = xr.DataArray(
        data=phi_array.T,  # transpose so dims are (phi, r)
        dims=["phi", "r"],
        coords={
            "phi": np.arange(len((BC.V_0))),  # or V_0.index if it's a Series
            "r": BC.r_s
        },
        name="FlowAngle (phi)[deg]"
    )
    
    return phi_xr