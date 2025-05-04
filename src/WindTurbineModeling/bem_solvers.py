#TODO: Add docstrings to all functions and classes

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
    """Base class containing common BEM solver functionality"""
    def __init__(self):
        """
        Initializes an instance of the class.

        Attributes:
            results (None): Placeholder for storing computation results.
            elemental_data (None): Placeholder for storing data related to elements.
            inputs (None): Placeholder for storing input parameters.
        """
        self.results = None
        self.elemental_data = None
        self.inputs = None

    def load_input_data(self):
        """
        Load all required input data for the wind turbine blade element momentum (BEM) solver.
        This method loads and processes the following input data:
        - Blade geometry data from a specified input file, filtering out invalid blade IDs (BlAFID <= 1).
        - Operational settings from a specified input file, filtering wind speeds to the range [3.0, 25.0].
        - Airfoil geometry and aerodynamic coefficient data from specified files, categorized by shape and coefficient identifiers.
        The processed data is stored in the `self.inputs` dictionary with the following keys:
        - 'blade_data': Filtered blade geometry data as a DataFrame.
        - 'settings': Filtered operational settings as a DataFrame.
        - 'airfoil_shapes': List of airfoil shape geometries.
        - 'airfoils': Dictionary mapping valid blade IDs (BlAFIDs) to their corresponding airfoil coefficient DataFrames.
        - 'valid_blafids': List of valid blade IDs (BlAFIDs).
        Returns:
            dict: A dictionary containing all processed input data.
        """
        """Load all required input data"""
        blade_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
        blade_data = blade_data[blade_data["BlAFID"] > 1].reset_index(drop=True)

        settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)
        settings = settings[(settings["WindSpeed"] >= 3.0) & (settings["WindSpeed"] <= 25.0)]

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

    def solve_blade_element(self, V0, omega, r, pitch, beta, chord, df_polar, num_blades, radius, rho, max_iter, tolerance, af_id):
        """
        Solve for a single blade element using the Blade Element Momentum (BEM) method.
        Parameters:
        -----------
        V0 : float
            Free-stream wind velocity (m/s).
        omega : float
            Rotational speed of the turbine (rad/s).
        r : float
            Radial position of the blade element (m).
        pitch : float
            Blade pitch angle (rad).
        beta : float
            Twist angle of the blade element (rad).
        chord : float
            Chord length of the blade element (m).
        df_polar : pandas.DataFrame
            Airfoil polar data containing lift and drag coefficients.
        num_blades : int
            Number of blades in the turbine.
        radius : float
            Radius of the turbine rotor (m).
        rho : float
            Air density (kg/m^3).
        max_iter : int
            Maximum number of iterations for convergence.
        tolerance : float
            Convergence tolerance for induction factors.
        af_id : int
            Airfoil identifier for the blade element.
        Returns:
        --------
        dict or None
            A dictionary containing the following keys if the solution converges:
            - 'p_n': float, Normal force per unit length (N/m).
            - 'p_t': float, Tangential force per unit length (N/m).
            - 'elemental': dict, Detailed results for the blade element, including:
                - 'V0': float, Free-stream wind velocity (m/s).
                - 'r': float, Radial position of the blade element (m).
                - 'alpha': float, Angle of attack (rad).
                - 'Cl': float, Lift coefficient.
                - 'Cd': float, Drag coefficient.
                - 'a': float, Axial induction factor.
                - 'a_prime': float, Tangential induction factor.
                - 'phi': float, Flow angle (rad).
                - 'F': float, Prandtl tip loss factor.
                - 'p_n': float, Normal force per unit length (N/m).
                - 'p_t': float, Tangential force per unit length (N/m).
                - 'sigma': float, Local solidity.
                - 'v_rel': float, Relative velocity at the blade element (m/s).
                - 'beta': float, Twist angle of the blade element (rad).
                - 'chord': float, Chord length of the blade element (m).
                - 'af_id': int, Airfoil identifier for the blade element.
            Returns None if the solution does not converge or if invalid results are encountered.
        """
        a, a_prime = 0.0, 0.0
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

        v_rel, p_n, p_t = calc_relative_velocity_and_forces(V0, omega, r, a, a_prime, rho, chord, Cn, Ct)

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
        """
        Prepare data for plotting from the solver results.
        This method processes the results of the solver, organizes them into a 
        dictionary, and converts units where necessary for better visualization.
        Returns:
            dict: A dictionary containing the following keys:
                - 'wind_speeds' (numpy.ndarray): Array of wind speeds (V_0) from the results.
                - 'power' (numpy.ndarray): Array of power values (P) in megawatts.
                - 'thrust' (numpy.ndarray): Array of thrust values (T) in kilonewtons.
                - 'cp' (numpy.ndarray): Array of power coefficients (C_P).
                - 'ct' (numpy.ndarray): Array of thrust coefficients (C_T).
                - 'pitch' (numpy.ndarray, optional): Array of pitch values, if available in the results.
                - 'rpm' (numpy.ndarray, optional): Array of RPM values, if available in the results.
        Raises:
            ValueError: If the solver has not been run and the 'results' attribute is missing.
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
        """
        Save results, and airfoil data to CSV files.
        Parameters:
        -----------
        results : list or pandas.DataFrame
            A collection of performance summary data to be saved.
        elemental_data : list or pandas.DataFrame
            Elemental data to be saved.
        airfoil_data : dict
            A dictionary where keys are airfoil IDs and values are pandas.DataFrame
            objects containing airfoil-specific data.
        prefix : str, optional
            A prefix to prepend to the output file names (default is an empty string).
        Outputs:
        --------
        - Saves a performance summary CSV file in the "results" directory.
        - Saves an elemental data CSV file in the "results" directory.
        - Saves individual airfoil data CSV files in the "results/airfoils" directory.
        """
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        pd.DataFrame(results).sort_values("V_0").to_csv(output_dir / f"{prefix}performance_summary.csv", index=False)
        pd.DataFrame(elemental_data).to_csv(output_dir / f"{prefix}elemental_data.csv", index=False)

        airfoil_dir = output_dir / "airfoils"
        airfoil_dir.mkdir(exist_ok=True)
        for af_id, df in airfoil_data.items():
            df.to_csv(airfoil_dir / f"airfoil_{af_id}.csv", index=False)

class BEMSolver(BaseBEMSolver):
    """
    Standard BEM (Blade Element Momentum) Solver for wind turbine modeling.
    This class provides methods to perform BEM calculations, process turbine data, 
    and compute aerodynamic performance metrics for wind turbines.
    Methods:

    """

    def run(self):
        """
        Executes the BEM calculation workflow, including loading input data, 
        performing calculations, saving results, and preparing data for plotting.
        perform_bem_calculations(inputs):
        Performs BEM calculations for multiple wind speed and turbine settings.
        Parameters:
            inputs (dict): A dictionary containing turbine data, settings, 
                - blade geometry, airfoil data, and other required inputs.
        Returns:
            tuple: A tuple containing:
                - summary_results (list): A list of dictionaries summarizing 
                    the results for each wind speed condition.
                - elemental_data (defaultdict): A dictionary containing 
                    elemental data (e.g., forces, moments) for each blade element.
        """

        print("1. Loading and parsing the provided turbine data...")
        inputs = self.load_input_data()

        print("2. Performing BEM calculations:")
        summary_results, elemental_data = self.perform_bem_calculations(inputs)

        print("3. Saving results...")
        self.save_results(summary_results, elemental_data, inputs['airfoils'])

        print("4. Preparing data for plotting...")

    def perform_bem_calculations(self, inputs):
        """
        Perform Blade Element Momentum (BEM) calculations for a range of wind speeds 
        and operating conditions.
        Parameters:
            inputs (dict): A dictionary containing the following keys:
                - 'blade_data' (pd.DataFrame): DataFrame containing blade geometry 
                  and aerodynamic properties (e.g., spanwise positions, chord lengths).
                - 'settings' (pd.DataFrame): DataFrame containing operating conditions 
                  (e.g., wind speed, pitch angle, rotational speed).
                - 'valid_blafids' (list): List of valid blade aerodynamic identifiers.
                - 'airfoils' (dict): Dictionary mapping airfoil identifiers to their 
                  aerodynamic data.
        Returns:
            tuple: A tuple containing:
                - summary_results (list): A list of dictionaries summarizing the 
                  results for each operating condition.
                - elemental_data (defaultdict): A defaultdict containing lists of 
                  elemental data for each spanwise position.
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
                V0, pitch, omega, r_s, blade_data, inputs['valid_blafids'], inputs['airfoils']
            )
            summary_results.append(condition_results['summary'])
            for key, values in condition_results['elemental'].items():
                elemental_data[key].extend(values)

        self.results = summary_results
        self.elemental_data = elemental_data
        return summary_results, elemental_data

    def calculate_single_condition(self, V0, pitch, omega, r_s, blade_data, valid_blafids, airfoils):
        """
        Calculates BEM results for a single wind speed and turbine condition.
        Parameters:
            V0 (float): Freestream wind speed in m/s.
            pitch (float): Blade pitch angle in degrees.
            omega (float): Rotor angular velocity in rad/s.
            r_s (array-like): Radial positions along the blade span.
            blade_data (DataFrame): DataFrame containing blade geometry data.
            valid_blafids (list): List of valid blade airfoil IDs.
            airfoils (list): List of airfoil polar data corresponding to blade airfoil IDs.
        Returns:
            dict: A dictionary containing:
                - 'summary': A dictionary summarizing the condition results, 
                    including thrust, torque, power, and performance coefficients.
                - 'elemental': A dictionary containing elemental data for 
                    each blade element.
        """
        elemental = defaultdict(list)
        dT_list, dM_list, r_used = [], [], []

        for r, af_id in zip(r_s, blade_data['BlAFID']):
            beta, chord = interpolate_blade_geometry(r, blade_data['BlSpn'], blade_data['BlTwist'], blade_data['BlChord'])
            df_polar = airfoils[valid_blafids.index(af_id)]
            bem_result = self.solve_blade_element(V0, omega, r, pitch, beta, chord, df_polar, NUMBER_BLADES, R, RHO, MAX_ITER, TOLERANCE, af_id)

            if bem_result:
                for key, value in bem_result['elemental'].items():
                    elemental[key].append(value)
                dT_list.append(NUMBER_BLADES * bem_result['p_n'])
                dM_list.append(NUMBER_BLADES * r * bem_result['p_t'])
                r_used.append(r)

        T, M, P_total, CP, CT = compute_totals_and_coefficients(dT_list, dM_list, omega, r_used, RHO, R, V0, RATED_POWER)

        return {
            'summary': {
                "V_0": V0, "T": T, "M": M, "P": P_total,
                "C_P": CP, "C_T": CT, "omega": omega, "pitch": pitch
            },
            'elemental': elemental
        }

class BEMSolverOpt(BaseBEMSolver):
    """
    Optimal Control BEM Solver.
    This class implements an optimal control strategy for solving the 
    Blade Element Momentum (BEM) equations for wind turbine performance 
    analysis.     
    """
    def __init__(self):
        super().__init__()
        self.pitch_interp = None
        self.rpm_interp = None
        self.min_ws = None
        self.max_ws = None

    def run(self):
        """
        Executes the main workflow for the BEM (Blade Element Momentum) solver.
        This method performs the following steps:
        1. Loads and parses the provided turbine data.
        2. Sets up the optimal control strategy for the turbine.
        3. Performs BEM calculations using the optimal pitch and RPM values.
        4. Saves the calculated results to files.
        5. Prepares data for plotting.
        Input:
            None (Relies on instance attributes and methods to load input data and configurations).
        Output:
            None (Results are saved to files and prepared for visualization).
        """
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
        """
        Sets up the optimal operational strategy for the wind turbine by interpolating 
        pitch angles and rotational speeds based on wind speed data.
        Parameters:
        -----------
        inputs : dict
            A dictionary containing the following key:
            - 'settings': A pandas DataFrame with columns ['WindSpeed', 'PitchAngle', 'RotSpeed'].
              - 'WindSpeed': Array-like, wind speed values.
              - 'PitchAngle': Array-like, pitch angle values.
              - 'RotSpeed': Array-like, rotational speed (RPM) values.
        Attributes Set:
        ----------------
        pitch_interp : scipy.interpolate.interp1d
            Interpolation function for pitch angle based on wind speed.
        rpm_interp : scipy.interpolate.interp1d
            Interpolation function for rotational speed (RPM) based on wind speed.
        min_ws : float
            Minimum wind speed value from the input data.
        max_ws : float
            Maximum wind speed value from the input data.
        """
        operational_data = inputs['settings'][['WindSpeed', 'PitchAngle', 'RotSpeed']].values
        wind_speeds = operational_data[:, 0]
        pitches = savgol_filter(operational_data[:, 1], 5, 2)
        rpms = operational_data[:, 2]

        self.pitch_interp = interp1d(wind_speeds, pitches, kind='linear', fill_value="extrapolate")
        self.rpm_interp = interp1d(wind_speeds, rpms, kind='linear', fill_value="extrapolate")
        self.min_ws, self.max_ws = min(wind_speeds), max(wind_speeds)

    def generate_wind_speed_range(self):
        """
        Generates a range of wind speeds by concatenating three linearly spaced arrays.
        Returns:
            numpy.ndarray: A concatenated array of wind speeds covering the specified ranges.
        """
        return np.concatenate([
            np.linspace(self.min_ws, 10.0, 15),
            np.linspace(10.1, 11.5, 10),
            np.linspace(11.6, self.max_ws, 15)
        ])

    def perform_optimal_calculations(self, inputs):
        """
        Perform optimal calculations for a range of wind speeds using optimal control.
        Parameters:
        -----------
        inputs : dict
            A dictionary containing the following keys:
            - 'blade_data': DataFrame containing blade geometry and aerodynamic properties.
            - 'valid_blafids': List of valid blade aerodynamic identifiers.
            - 'airfoils': Dictionary of airfoil data.
        Returns:
        --------
        tuple
            A tuple containing:
            - summary_results (list): A list of dictionaries summarizing the results for each wind speed.
              Each dictionary contains keys such as 'RPM', 'Pitch', and other summary metrics.
            - elemental_data (defaultdict): A defaultdict containing lists of elemental data
              for each wind speed. Keys correspond to data types (e.g., forces, moments).
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

    def calculate_single_condition(self, V0, pitch, omega, r_s, blade_data, valid_blafids, airfoils):
        """
        Calculate aerodynamic performance for a single operating condition of a wind turbine.
        Parameters:
        -----------
        V0 : float
            Free-stream wind speed (m/s).
        pitch : float
            Blade pitch angle (degrees).
        omega : float
            Rotor angular velocity (rad/s).
        r_s : list of float
            Radial positions along the blade span (m).
        blade_data : dict
            Dictionary containing blade geometry data with keys:
            - 'BlSpn': list of float, spanwise positions (m).
            - 'BlTwist': list of float, twist angles (degrees).
            - 'BlChord': list of float, chord lengths (m).
            - 'BlAFID': list of int, airfoil IDs for each spanwise position.
        valid_blafids : list of int
            List of valid airfoil IDs.
        airfoils : list of pandas.DataFrame
            List of airfoil polar data corresponding to valid_blafids.
        Returns:
        --------
        dict
            A dictionary containing:
            - 'summary': dict with keys:
            - "V_0": float, free-stream wind speed (m/s).
            - "T": float, total thrust (N).
            - "M": float, total torque (Nm).
            - "P": float, total power (W).
            - "C_P": float, power coefficient.
            - "C_T": float, thrust coefficient.
            - "omega": float, rotor angular velocity (rad/s).
            - "pitch": float, blade pitch angle (degrees).
            - 'elemental': defaultdict of lists containing elemental aerodynamic data along the blade span.
        """
        elemental = defaultdict(list)
        dT_list, dM_list, r_used = [], [], []

        for r, af_id in zip(r_s, blade_data['BlAFID']):
            beta, chord = interpolate_blade_geometry(r, blade_data['BlSpn'], blade_data['BlTwist'], blade_data['BlChord'])
            df_polar = airfoils[valid_blafids.index(af_id)]
            bem_result = self.solve_blade_element(V0, omega, r, pitch, beta, chord, df_polar,
                                                NUMBER_BLADES, R, RHO, MAX_ITER, TOLERANCE, af_id)

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
