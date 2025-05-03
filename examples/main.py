"""
main.py - Interactive entry point for the WindTurbineModeling package.

This script launches an interactive interface allowing users to run either a standard or
optimal control Blade Element Momentum (BEM) simulation for wind turbine performance analysis.

How to Run:
-----------
From the terminal or command line, execute:
    python examples/main.py 

Workflow:
---------
1. The user is prompted to choose a solver type:
    [1] Standard BEM Solver
        - Uses static pitch and RPM from predefined operational settings.
    [2] Optimal Control BEM Solver
        - Uses interpolated pitch and RPM curves for adaptive control.
2. The solver loads all necessary data: blade geometry, airfoil profiles, and performance conditions.
3. Blade element calculations are performed across a range of wind speeds.
4. Results (thrust, torque, power, coefficients) are stored and visualized interactively.
5. Figures can be saved, and the solver can be re-run from within the interface.
6. The user can exit the interface at any time.

Core Functional Capabilities (Requirement Coverage):
----------------------------------------------------
This package fulfills all required functionality for a wind turbine blade element momentum (BEM) solver:

1. Load and parse the provided turbine data
    - Geometry, airfoil profiles, and operating conditions are automatically imported.

2. Plot the provided airfoil shapes
    - `plot_airfoil_shapes()` renders all loaded profiles in a single figure.

3. Compute Cl and Cd vs span position (r) and angle of attack (α)
    - Elemental polar data is extracted and plotted using:
        - `plot_cl_cd_vs_span()`
        - `plot_cl_cd_vs_alpha()`

4. Compute axial (a) and tangential (a') induction factors
    - Varies with span (r), inflow speed (V₀), pitch (θₚ), and ω.
    - Plotted using:
        - `plot_induction_vs_span()`
        - `plot_induction_vs_v0()`
        - `plot_induction_vs_pitch()`
        - `plot_induction_vs_omega()`

5. Compute thrust (T), torque (M), and power (P)
    - All derived as functions of V₀, θₚ, and ω across operating conditions.

6. Compute optimal pitch (θₚ) and rotational speed (ω)
    - Based on interpolated strategy from `IEA_15MW_RWT_Onshore.opt`.

7. Compute and plot power curve P(V₀) and thrust curve T(V₀)
    - Shown via:
        - `plot_power_curve()`
        - `plot_thrust_curve()`

Bonus Features (Extra Functionalities):
---------------------------------------
- Plot a schematic diagram of the turbine rotor:
    - `plot_wind_turbine()`

- Plot Prandtl Tip Loss Factor (F) vs span:
    - `plot_tip_loss_vs_span()`

- Plot power coefficient (Cp) and thrust coefficient (Ct) surfaces as functions of:
    - Blade pitch angle (θₚ)
    - Tip speed ratio (λ)
    - via `plot_cp_ct_surfaces()`

Author: [Hannah, Lenssa, Dorian]
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from WindTurbineModeling.interactive import main

if __name__ == "__main__":
    main()