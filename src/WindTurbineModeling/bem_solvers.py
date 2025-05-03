"""Wind Turbine Blade Element Momentum (BEM) Solver Module.

This module implements both standard and optimal control BEM solvers
for wind turbine performance analysis.
"""

import sys
import os
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Configure module import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from WindTurbineModeling.read import get_files_by_extension
    from WindTurbineModeling.load import (
        load_blade_geometry,
        load_operational_settings,
        load_geometry,
        load_airfoil_coefficients
    )
    from WindTurbineModeling.equations import (
        calc_omega,
        calc_flow_angle,
        calc_local_angle_of_attack,
        calc_local_lift_drag_force,
        calc_normal_tangential_constants,
        calc_local_solidity,
        calc_prandtl_tip_loss,
        update_induction_factors,
        calc_relative_velocity_and_forces,
        compute_totals_and_coefficients,
        interpolate_blade_geometry
    )
    from WindTurbineModeling.plot import (
        plot_power_curve,
        plot_cp_curve,
        plot_ct_curve,
        plot_thrust_curve,
        plot_airfoil_shapes,
        plot_wind_turbine
    )
    from WindTurbineModeling.config import (
        BLADE_DEFINITION_INPUT_FILE_PATH,
        OPERATIONAL_CONDITIONS_FILE_PATH,
        AIRFOIL_DATA,
        AIRFOIL_SHAPE_IDENTIFIER,
        AIRFOIL_INFO_IDENTIFIER,
        NUMBER_BLADES,
        R,
        RHO,
        MAX_ITER,
        TOLERANCE,
        RATED_POWER
    )
except ImportError as e:
    print(f"Import Error: {e}")
    raise

class BaseBEMSolver:
    """Base class for Blade Element Momentum (BEM) solvers.
    
    Provides common functionality for both standard and optimal BEM solvers.
    """
    
    def __init__(self):
        """Initialize the BEM solver with default parameters."""
        self.results = None
        self.elemental_data = None
        self.inputs = None

    def load_input_data(self):
        """Load and parse turbine input data from configuration files.
        
        Returns:
            dict: Dictionary containing:
                - blade_data: DataFrame of blade geometry
                - settings: DataFrame of operational conditions
                - airfoil_shapes: Dictionary of airfoil geometries
                - airfoils: Dictionary of airfoil coefficients
                - valid_blafids: List of valid airfoil IDs
        """
        blade_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
        blade_data = blade_data[blade_data["BlAFID"] > 1].reset_index(drop=True)

        settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)
        settings = settings[(settings["WindSpeed"] >= 3.0) & 
                           (settings["WindSpeed"] <= 25.0)]

        airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])
        airfoil_shapes = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
        airfoil_coeffs = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]

        shapes = load_geometry(airfoil_shapes)
        _, airfoils_raw = load_airfoil_coefficients(airfoil_coeffs)

        valid_blafids, airfoils = [], []
        for i, df in enumerate(airfoils_raw):
            valid_blafids.append(i + 1)
            airfoils.append(df)

        self.inputs = {
            'blade_data': blade_data,
            'settings': settings,
            'airfoil_shapes': shapes,
            'airfoils': dict(zip(valid_blafids, airfoils)),
            'valid_blafids': valid_blafids
        }
        return self.inputs

    def solve_blade_element(self, V0, omega, r, pitch, beta, chord, df_polar, 
                          num_blades, radius, rho, max_iter, tolerance, af_id):
        """Solve BEM equations for a single blade element.
        
        Args:
            V0 (float): Wind speed (m/s)
            omega (float): Rotational speed (rad/s)
            r (float): Radial position (m)
            pitch (float): Pitch angle (deg)
            beta (float): Twist angle (deg)
            chord (float): Chord length (m)
            df_polar (DataFrame): Airfoil polar data
            num_blades (int): Number of blades
            radius (float): Rotor radius (m)
            rho (float): Air density (kg/m³)
            max_iter (int): Maximum iterations
            tolerance (float): Convergence tolerance
            af_id (int): Airfoil ID
            
        Returns:
            dict: Dictionary containing:
                - p_n: Normal force (N)
                - p_t: Tangential force (N)
                - elemental: Dictionary of elemental results
            or None if solution fails
        """
        a, a_prime = 0.0, 0.0  # Initial induction factors

        for _ in range(max_iter):
            phi = calc_flow_angle(V0, omega, r, a, a_prime)
            alpha = calc_local_angle_of_attack(phi, pitch, beta)
            Cl, Cd = calc_local_lift_drag_force(alpha, df_polar)
            Cn, Ct = calc_normal_tangential_constants(phi, Cl, Cd)
            sigma = calc_local_solidity(num_blades, chord, r)
            F = calc_prandtl_tip_loss(num_blades, radius, r, phi)

            a_new, a_prime_new = update_induction_factors(phi, sigma, Cn, Ct, F, a)
            if abs(a_new - a) < tolerance and abs(a_prime_new - a_prime) < tolerance:
                break
            a, a_prime = a_new, a_prime_new

        v_rel, p_n, p_t = calc_relative_velocity_and_forces(
            V0, omega, r, a, a_prime, rho, chord, Cn, Ct)

        if np.isnan(Cl) or np.isnan(Cd) or np.isnan(p_t) or Cl == 0 or p_t < 0:
            return None

        return {
            'p_n': p_n,
            'p_t': p_t,
            'elemental': {
                'V0': V0, 'r': r, 'alpha': alpha, 'Cl': Cl, 'Cd': Cd,
                'a': a, 'a_prime': a_prime, 'phi': phi, 'F': F,
                'p_n': p_n, 'p_t': p_t, 'sigma': sigma, 'v_rel': v_rel,
                'beta': beta, 'chord': chord, 'af_id': af_id
            }
        }

    def get_plot_data(self):
        """Prepare results data for visualization.
        
        Returns:
            dict: Dictionary containing:
                - wind_speeds: Array of wind speeds
                - power: Array of power values (MW)
                - thrust: Array of thrust values (kN)
                - cp: Array of power coefficients
                - ct: Array of thrust coefficients
                - pitch: Array of pitch angles (if available)
                - rpm: Array of RPM values (if available)
                
        Raises:
            ValueError: If solver hasn't been run yet
        """
        if not hasattr(self, 'results'):
            raise ValueError("Run the solver first.")

        df = pd.DataFrame(self.results).sort_values("V_0")
        data = {
            'wind_speeds': df["V_0"].values,
            'power': df["P"].values / 1e6,
            'thrust': df["T"].values / 1000,
            'cp': df["C_P"].values,
            'ct': df["C_T"].values
        }
        if "Pitch" in df.columns:
            data['pitch'] = df["Pitch"].values
        if "RPM" in df.columns:
            data['rpm'] = df["RPM"].values
        return data

    def save_results(self, results, elemental_data, airfoil_data, prefix=""):
        """Save solver results to CSV files.
        
        Args:
            results (list): Summary results
            elemental_data (dict): Elemental force data
            airfoil_data (dict): Airfoil coefficient data
            prefix (str): Optional filename prefix
        """
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        pd.DataFrame(results).sort_values("V_0").to_csv(
            output_dir / f"{prefix}performance_summary.csv", index=False)
        pd.DataFrame(elemental_data).to_csv(
            output_dir / f"{prefix}elemental_data.csv", index=False)

        airfoil_dir = output_dir / "airfoils"
        airfoil_dir.mkdir(exist_ok=True)
        for af_id, df in airfoil_data.items():
            df.to_csv(airfoil_dir / f"airfoil_{af_id}.csv", index=False)


class BEMSolver(BaseBEMSolver):
    """Standard BEM Solver implementation.
    
    Solves the BEM equations using fixed operational settings.
    """
    
    def run(self):
        """Execute the complete BEM analysis workflow."""
        print("1. Loading and parsing the provided turbine data...")
        inputs = self.load_input_data()

        print("2. Performing BEM calculations:")
        summary_results, elemental_data = self.perform_bem_calculations(inputs)

        print("3. Saving results...")
        self.save_results(summary_results, elemental_data, inputs['airfoils'])

        print("4. Preparing data for plotting...")

    def perform_bem_calculations(self, inputs):
        """Perform BEM calculations for all operational conditions.
        
        Args:
            inputs (dict): Input data dictionary
            
        Returns:
            tuple: (summary_results, elemental_data)
        """
        summary_results = []
        elemental_data = defaultdict(list)

        blade_data = inputs['blade_data']
        settings = inputs['settings']
        r_s = blade_data['BlSpn'].values

        for i, row in enumerate(settings.itertuples(), start=1):
            V0 = row.WindSpeed
            print(f"   → [{i}/{len(settings)}] Solving for wind speed: {V0:.1f} m/s")

            pitch = row.PitchAngle
            omega = calc_omega(row.RotSpeed)

            condition_results = self.calculate_single_condition(
                V0, pitch, omega, r_s, blade_data, 
                inputs['valid_blafids'], inputs['airfoils']
            )
            summary_results.append(condition_results['summary'])
            for key, values in condition_results['elemental'].items():
                elemental_data[key].extend(values)

        self.results = summary_results
        self.elemental_data = elemental_data
        return summary_results, elemental_data

    def calculate_single_condition(self, V0, pitch, omega, r_s, blade_data, 
                                 valid_blafids, airfoils):
        """Calculate BEM solution for a single wind speed condition.
        
        Args:
            V0 (float): Wind speed (m/s)
            pitch (float): Pitch angle (deg)
            omega (float): Rotational speed (rad/s)
            r_s (ndarray): Array of radial positions
            blade_data (DataFrame): Blade geometry data
            valid_blafids (list): Valid airfoil IDs
            airfoils (dict): Airfoil coefficient data
            
        Returns:
            dict: Dictionary containing summary and elemental results
        """
        elemental = defaultdict(list)
        dT_list, dM_list, r_used = [], [], []

        for r, af_id in zip(r_s, blade_data['BlAFID']):
            beta, chord = interpolate_blade_geometry(
                r, blade_data['BlSpn'], 
                blade_data['BlTwist'], blade_data['BlChord']
            )
            df_polar = airfoils[valid_blafids.index(af_id)]
            bem_result = self.solve_blade_element(
                V0, omega, r, pitch, beta, chord, df_polar, 
                NUMBER_BLADES, R, RHO, MAX_ITER, TOLERANCE, af_id
            )

            if bem_result:
                for key, value in bem_result['elemental'].items():
                    elemental[key].append(value)
                dT_list.append(NUMBER_BLADES * bem_result['p_n'])
                dM_list.append(NUMBER_BLADES * r * bem_result['p_t'])
                r_used.append(r)

        T, M, P_total, CP, CT = compute_totals_and_coefficients(
            dT_list, dM_list, omega, r_used, RHO, R, V0, RATED_POWER)

        return {
            'summary': {
                "V_0": V0, "T": T, "M": M, "P": P_total,
                "C_P": CP, "C_T": CT, "omega": omega, "pitch": pitch
            },
            'elemental': elemental
        }


class BEMSolverOpt(BaseBEMSolver):
    """Optimal Control BEM Solver implementation.
    
    Solves the BEM equations using optimal pitch and RPM strategies.
    """
    
    def __init__(self):
        """Initialize the optimal BEM solver."""
        super().__init__()
        self.pitch_interp = None
        self.rpm_interp = None
        self.min_ws = None
        self.max_ws = None

    def run(self):
        """Execute the complete optimal BEM analysis workflow."""
        print("1. Loading and parsing the provided turbine data...")
        inputs = self.load_input_data()

        print("2. Setting up optimal control strategy...")
        self.setup_optimal_strategy(inputs)

        print("3. Performing BEM calculations using optimal pitch/RPM...")
        summary_results, elemental_data = self.perform_optimal_calculations(inputs)

        print("4. Saving results...")
        self.save_results(summary_results, elemental_data, inputs['airfoils'], "optimal_")

        print("5. Preparing data for plotting...")

    def setup_optimal_strategy(self, inputs):
        """Initialize optimal pitch and RPM interpolation functions.
        
        Args:
            inputs (dict): Input data dictionary
        """
        operational_data = inputs['settings'][['WindSpeed', 'PitchAngle', 'RotSpeed']].values
        wind_speeds = operational_data[:, 0]
        pitches = savgol_filter(operational_data[:, 1], 5, 2)
        rpms = operational_data[:, 2]

        self.pitch_interp = interp1d(wind_speeds, pitches, kind='linear', fill_value="extrapolate")
        self.rpm_interp = interp1d(wind_speeds, rpms, kind='linear', fill_value="extrapolate")
        self.min_ws, self.max_ws = min(wind_speeds), max(wind_speeds)

    def generate_wind_speed_range(self):
        """Generate a non-linear wind speed range for analysis.
        
        Returns:
            ndarray: Array of wind speeds with higher resolution near rated speed
        """
        return np.concatenate([
            np.linspace(self.min_ws, 10.0, 15),
            np.linspace(10.1, 11.5, 10),  # Higher resolution near rated speed
            np.linspace(11.6, self.max_ws, 15)
        ])

    def perform_optimal_calculations(self, inputs):
        """Perform BEM calculations using optimal control strategy.
        
        Args:
            inputs (dict): Input data dictionary
            
        Returns:
            tuple: (summary_results, elemental_data)
        """
        summary_results = []
        elemental_data = defaultdict(list)
        wind_speed_range = self.generate_wind_speed_range()

        for i, V0 in enumerate(wind_speed_range, start=1):
            print(f"   → [{i}/{len(wind_speed_range)}] Solving for V₀ = {V0:.2f} m/s using optimal control...")

            pitch = float(self.pitch_interp(V0))
            rpm = float(self.rpm_interp(V0))
            omega = calc_omega(rpm)

            condition_results = self.calculate_single_condition(
                V0, pitch, omega,
                inputs['blade_data']['BlSpn'].values,
                inputs['blade_data'],
                inputs['valid_blafids'],
                inputs['airfoils']
            )
            summary_results.append({**condition_results['summary'], "RPM": rpm, "Pitch": pitch})
            for key, values in condition_results['elemental'].items():
                elemental_data[key].extend(values)

        self.results = summary_results
        self.elemental_data = elemental_data
        return summary_results, elemental_data

    def calculate_single_condition(self, V0, pitch, omega, r_s, blade_data, 
                                 valid_blafids, airfoils):
        """Calculate BEM solution for a single wind speed condition (optimal version).
        
        Args:
            V0 (float): Wind speed (m/s)
            pitch (float): Pitch angle (deg)
            omega (float): Rotational speed (rad/s)
            r_s (ndarray): Array of radial positions
            blade_data (DataFrame): Blade geometry data
            valid_blafids (list): Valid airfoil IDs
            airfoils (dict): Airfoil coefficient data
            
        Returns:
            dict: Dictionary containing summary and elemental results
        """
        elemental = defaultdict(list)
        dT_list, dM_list, r_used = [], [], []

        for r, af_id in zip(r_s, blade_data['BlAFID']):
            beta, chord = interpolate_blade_geometry(
                r, blade_data['BlSpn'], 
                blade_data['BlTwist'], blade_data['BlChord']
            )
            df_polar = airfoils[valid_blafids.index(af_id)]
            bem_result = self.solve_blade_element(
                V0, omega, r, pitch, beta, chord, df_polar,
                NUMBER_BLADES, R, RHO, MAX_ITER, TOLERANCE, af_id
            )

            if bem_result:
                for key, value in bem_result['elemental'].items():
                    elemental[key].append(value)
                dT_list.append(NUMBER_BLADES * bem_result['p_n'])
                dM_list.append(NUMBER_BLADES * r * bem_result['p_t'])
                r_used.append(r)

        T, M, P_total, CP, CT = compute_totals_and_coefficients(
            dT_list, dM_list, omega, r_used, RHO, R, V0, RATED_POWER
        )

        return {
            'summary': {
                "V_0": V0, "T": T, "M": M, "P": P_total,
                "C_P": CP, "C_T": CT, "omega": omega, "pitch": pitch
            },
            'elemental': elemental
        }