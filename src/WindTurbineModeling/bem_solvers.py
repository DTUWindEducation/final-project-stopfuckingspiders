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
        self.results = None
        self.elemental_data = None
        self.inputs = None

    def load_input_data(self):
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
        """Solve for a single blade element"""
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
        """Prepare data for plotting"""
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
        """Save results to CSV files"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        pd.DataFrame(results).sort_values("V_0").to_csv(output_dir / f"{prefix}performance_summary.csv", index=False)
        pd.DataFrame(elemental_data).to_csv(output_dir / f"{prefix}elemental_data.csv", index=False)

        airfoil_dir = output_dir / "airfoils"
        airfoil_dir.mkdir(exist_ok=True)
        for af_id, df in airfoil_data.items():
            df.to_csv(airfoil_dir / f"airfoil_{af_id}.csv", index=False)

class BEMSolver(BaseBEMSolver):
    """Standard BEM Solver"""
    def run(self):
        start_time = time()
        inputs = self.load_input_data()
        summary_results, elemental_data = self.perform_bem_calculations(inputs)
        self.save_results(summary_results, elemental_data, inputs['airfoils'])

    def perform_bem_calculations(self, inputs):
        summary_results = []
        elemental_data = defaultdict(list)

        blade_data = inputs['blade_data']
        settings = inputs['settings']
        r_s = blade_data['BlSpn'].values

        for _, row in settings.iterrows():
            V0 = row['WindSpeed']
            pitch = row['PitchAngle']
            omega = calc_omega(row['RotSpeed'])

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
    """Optimal Control BEM Solver"""
    def __init__(self):
        super().__init__()
        self.pitch_interp = None
        self.rpm_interp = None
        self.min_ws = None
        self.max_ws = None

    def run(self):
        start_time = time()
        inputs = self.load_input_data()
        self.setup_optimal_strategy(inputs)
        summary_results, elemental_data = self.perform_optimal_calculations(inputs)
        self.save_results(summary_results, elemental_data, inputs['airfoils'], "optimal_")

    def setup_optimal_strategy(self, inputs):
        operational_data = inputs['settings'][['WindSpeed', 'PitchAngle', 'RotSpeed']].values
        wind_speeds = operational_data[:, 0]
        pitches = savgol_filter(operational_data[:, 1], 5, 2)
        rpms = operational_data[:, 2]

        self.pitch_interp = interp1d(wind_speeds, pitches, kind='linear', fill_value="extrapolate")
        self.rpm_interp = interp1d(wind_speeds, rpms, kind='linear', fill_value="extrapolate")
        self.min_ws, self.max_ws = min(wind_speeds), max(wind_speeds)

    def generate_wind_speed_range(self):
        return np.concatenate([
            np.linspace(self.min_ws, 10.0, 15),
            np.linspace(10.1, 11.5, 10),
            np.linspace(11.6, self.max_ws, 15)
        ])

    def perform_optimal_calculations(self, inputs):
        summary_results = []
        elemental_data = defaultdict(list)
        wind_speed_range = self.generate_wind_speed_range()

        for V0 in wind_speed_range:
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
