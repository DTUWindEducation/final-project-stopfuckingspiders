"""
WindTurbineModeling Package

This package provides a Blade Element Momentum (BEM) solver framework for
analyzing wind turbine performance,
including standard and optimal control strategies, data loading,
visualization, and aerodynamic calculations.

Available Submodules:
---------------------
- bem_solvers: BEMSolver and BEMSolverOpt classes
- load: Utilities for loading blade geometry, operational data, and airfoil
data
- equations: Core aerodynamic computation functions
- plot: Visualization utilities for performance metrics and turbine geometry
- interactive: CLI driver for simulation and plotting

Exposed API:
------------
This __init__.py exposes key classes and functions at the top level to
simplify imports.
"""

# Version
__version__ = "0.1.0"

# Expose solver interfaces
from .bem_solvers import BEMSolver, BEMSolverOpt

# Expose core utility modules as namespaces
from . import read
from . import load
from . import plot
from . import equations

# Public API (used when doing `from WindTurbineModeling import *`)
__all__ = [
    "BEMSolver",
    "BEMSolverOpt",
    "read",
    "load",
    "plot",
    "equations"
]
