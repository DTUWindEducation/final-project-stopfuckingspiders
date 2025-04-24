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
    print("Current Python path:")
    print(sys.path)
    raise

class BaseBEMSolver:
    """Base class containing common BEM solver functionality"""
    def __init__(self):
        self.results = None
        self.elemental_data = None
        self.inputs = None

    def load_input_data(self):
        """Load all required input data (common for both solvers)"""
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

    def solve_blade_element(self, V0, omega, r, pitch, beta, chord, 
                          df_polar, num_blades, radius, rho, max_iter, tolerance, af_id):
        """Core BEM solver for single blade element (common for both solvers)"""
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
        
        v_rel, p_n, p_t = calc_relative_velocity_and_forces(
            V0, omega, r, a, a_prime, rho, chord, Cn, Ct
        )
        
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
        """Returns structured data for plotting"""
        if not hasattr(self, 'results'):
            raise ValueError("Run the solver first with .run()")
        
        df = pd.DataFrame(self.results).sort_values("V_0")
        
        data = {
            'wind_speeds': df["V_0"].values,
            'power': df["P"].values / 1e6,  # Convert to MW
            'thrust': df["T"].values / 1000,  # Convert to kN
            'cp': df["C_P"].values,
            'ct': df["C_T"].values
        }
        
        # Add optimal-specific fields if available
        if hasattr(df, "Pitch"):
            data['pitch'] = df["Pitch"].values
        if hasattr(df, "RPM"):
            data['rpm'] = df["RPM"].values
            
        return data

    def save_results(self, results, elemental_data, airfoil_data, prefix=""):
        """Save results to files with optional prefix"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        summary_df = pd.DataFrame(results).sort_values("V_0")
        summary_df.to_csv(output_dir / f"{prefix}performance_summary.csv", index=False)
        
        elemental_df = pd.DataFrame(elemental_data)
        elemental_df.to_csv(output_dir / f"{prefix}elemental_data.csv", index=False)
        
        airfoil_dir = output_dir / "airfoils"
        airfoil_dir.mkdir(exist_ok=True)
        for af_id, df in airfoil_data.items():
            df.to_csv(airfoil_dir / f"airfoil_{af_id}.csv", index=False)

class BEMSolver(BaseBEMSolver):
    """Standard BEM solver implementation"""
    def run(self):
        start_time = time()
        inputs = self.load_input_data()
        summary_results, elemental_data = self.perform_bem_calculations(inputs)
        self.save_results(summary_results, elemental_data, inputs['airfoils'])
        self.generate_plots(summary_results, inputs['airfoil_shapes'])
        print(f"Standard BEM execution time: {time() - start_time:.2f} seconds")

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
                V0, pitch, omega, r_s, blade_data,
                inputs['valid_blafids'], inputs['airfoils']
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
            beta, chord = interpolate_blade_geometry(r, 
                                                   blade_data['BlSpn'],
                                                   blade_data['BlTwist'],
                                                   blade_data['BlChord'])
            
            df_polar = airfoils[valid_blafids.index(af_id)]
            bem_result = self.solve_blade_element(
                V0, omega, r, pitch, beta, chord, df_polar, NUMBER_BLADES,
                R, RHO, MAX_ITER, TOLERANCE, af_id
            )
            
            if bem_result is None:
                continue
                
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

    def generate_plots(self, summary_results, airfoil_shapes):
        summary_df = pd.DataFrame(summary_results).sort_values("V_0")
        V_arr = summary_df["V_0"].values
        P_arr = summary_df["P"].values / 1e6
        Cp_arr = summary_df["C_P"].values
        T_arr = summary_df["T"].values
        Ct_arr = summary_df["C_T"].values

        plot_power_curve(V_arr, P_arr, RATED_POWER)
        plot_cp_curve(V_arr, Cp_arr)
        plot_ct_curve(V_arr, Ct_arr)
        plot_thrust_curve(V_arr, T_arr)
        plot_airfoil_shapes(airfoil_shapes)
        plot_wind_turbine()
        plt.show()
    

class BEMSolverOpt(BaseBEMSolver):
    """BEM solver with optimal control strategy"""
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
        self.generate_plots(summary_results)
        print(f"Optimal BEM execution time: {time() - start_time:.2f} seconds")

    def setup_optimal_strategy(self, inputs):
        """Calculate optimal pitch and RPM strategy"""
        operational_data = inputs['settings'][['WindSpeed', 'PitchAngle', 'RotSpeed']].values
        wind_speeds = operational_data[:, 0]
        pitches = operational_data[:, 1]
        rpms = operational_data[:, 2]
        
        # Smooth the pitch angles for better interpolation
        pitches_smoothed = savgol_filter(pitches, window_length=5, polyorder=2)
        
        self.pitch_interp = interp1d(wind_speeds, pitches_smoothed, kind='linear', 
                                    fill_value=(pitches_smoothed[0], pitches_smoothed[-1]), 
                                    bounds_error=False)
        self.rpm_interp = interp1d(wind_speeds, rpms, kind='linear',
                                  fill_value=(rpms[0], rpms[-1]),
                                  bounds_error=False)
        self.min_ws, self.max_ws = min(wind_speeds), max(wind_speeds)

    def generate_wind_speed_range(self, step=0.1):
        """Generate wind speed range for optimal strategy"""
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
            
            summary_results.append({
                **condition_results['summary'],
                "RPM": rpm,
                "Pitch": pitch
            })
            for key, values in condition_results['elemental'].items():
                elemental_data[key].extend(values)
        
        self.results = summary_results
        self.elemental_data = elemental_data
        return summary_results, elemental_data

    def calculate_single_condition(self, V0, pitch, omega, r_s, blade_data, valid_blafids, airfoils):
        """Identical to BEMSolver's method but kept separate for flexibility"""
        elemental = defaultdict(list)
        dT_list, dM_list, r_used = [], [], []
        
        for r, af_id in zip(r_s, blade_data['BlAFID']):
            beta, chord = interpolate_blade_geometry(r, 
                                                   blade_data['BlSpn'],
                                                   blade_data['BlTwist'],
                                                   blade_data['BlChord'])
            
            df_polar = airfoils[valid_blafids.index(af_id)]
            bem_result = self.solve_blade_element(
                V0, omega, r, pitch, beta, chord, df_polar, NUMBER_BLADES,
                R, RHO, MAX_ITER, TOLERANCE, af_id
            )
            
            if bem_result is None:
                continue
                
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

    def generate_plots(self, summary_results):
        """Generate plots specific to optimal strategy"""
        summary_df = pd.DataFrame(summary_results).sort_values("V_0")
        V_arr = summary_df["V_0"].values
        P_arr = summary_df["P"].values / 1e6
        T_arr = summary_df["T"].values
        pitch_arr = summary_df["Pitch"].values
        rpm_arr = summary_df["RPM"].values

        # Create a 2x2 grid of plots
        plt.figure(figsize=(14, 10))
        
        # Power curve
        plt.subplot(2, 2, 1)
        plt.plot(V_arr, P_arr, 'b-', label='Optimal Power')
        plt.axhline(y=RATED_POWER/1e6, color='r', linestyle='--', label='Rated Power')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Power [MW]')
        plt.grid()
        plt.legend()
        plt.title('Optimal Power Curve')
        
        # Thrust curve
        plt.subplot(2, 2, 2)
        plt.plot(V_arr, T_arr/1000, 'g-', label='Thrust')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Thrust [kN]')
        plt.grid()
        plt.legend()
        plt.title('Optimal Thrust Curve')
        
        # Pitch control strategy
        plt.subplot(2, 2, 3)
        plt.plot(V_arr, pitch_arr, 'm-')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Pitch angle [deg]')
        plt.grid()
        plt.title('Pitch Control Strategy')
        
        # RPM control strategy
        plt.subplot(2, 2, 4)
        plt.plot(V_arr, rpm_arr, 'c-')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Rotational speed [RPM]')
        plt.grid()
        plt.title('RPM Control Strategy')
        
        plt.tight_layout()
        plt.show()

class BEMComparator:
    """Compares results between standard and optimal BEM solvers"""
    @staticmethod
    def compare_power(solver_std, solver_opt):
        """Plot power curve comparison"""
        data_std = solver_std.get_plot_data()
        data_opt = solver_opt.get_plot_data()
        
        plt.figure(figsize=(10, 6))
        plt.plot(data_std['wind_speeds'], data_std['power'], 'b-', label='Standard')
        plt.plot(data_opt['wind_speeds'], data_opt['power'], 'r--', label='Optimal')
        plt.axhline(y=RATED_POWER/1e6, color='k', linestyle=':', label='Rated Power')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power (MW)')
        plt.title('Power Curve Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def compare_thrust(solver_std, solver_opt):
        """Plot thrust curve comparison"""
        data_std = solver_std.get_plot_data()
        data_opt = solver_opt.get_plot_data()
        
        plt.figure(figsize=(10, 6))
        plt.plot(data_std['wind_speeds'], data_std['thrust'], 'b-', label='Standard')
        plt.plot(data_opt['wind_speeds'], data_opt['thrust'], 'r--', label='Optimal')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Thrust (kN)')
        plt.title('Thrust Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()
        

# Example usage when run directly
if __name__ == "__main__":
    # Run both solvers
    stanard = BEMSolver()
    print("Running standard BEM solver...")
    stanard.run()
    optimal = BEMSolverOpt()
    print("Running optimal BEM solver...")
    optimal.run()

    # Generate comparison plots
    BEMComparator.compare_power(stanard, optimal)
    print("Comparing thrust curves...") 
    BEMComparator.compare_thrust(stanard, optimal)

    # Optional: Show individual solver plots too
    stanard.generate_plots(pd.DataFrame(stanard.results), stanard.inputs['airfoil_shapes'])
    print("Generating plots for optimal BEM solver...")
    optimal.generate_plots(pd.DataFrame(optimal.results))