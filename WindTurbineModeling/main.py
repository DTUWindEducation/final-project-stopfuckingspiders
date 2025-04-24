import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.interpolate import interp1d
# Import your existing modules
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.equations import *
from WindTurbineModeling.plot import *
from config import (
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

def load_input_data():
    """Load all required input data"""
    # Blade geometry
    blade_data = load_blade_geometry(BLADE_DEFINITION_INPUT_FILE_PATH)
    blade_data = blade_data[blade_data["BlAFID"] > 1].reset_index(drop=True)

    # Operational conditions
    settings = load_operational_settings(OPERATIONAL_CONDITIONS_FILE_PATH)
    settings = settings[(settings["WindSpeed"] >= 3.0) & (settings["WindSpeed"] <= 25.0)]

    # Airfoil data
    airfoil_files = get_files_by_extension(AIRFOIL_DATA, [".dat", ".txt"])
    airfoil_shapes = [f for f in airfoil_files if AIRFOIL_SHAPE_IDENTIFIER in str(f)]
    airfoil_coeffs = [f for f in airfoil_files if AIRFOIL_INFO_IDENTIFIER in str(f)]
    shapes = load_geometry(airfoil_shapes)
    _, airfoils_raw = load_airfoil_coefficients(airfoil_coeffs)

    # Match airfoils to IDs
    valid_blafids, airfoils = [], []
    for i, df in enumerate(airfoils_raw):
        valid_blafids.append(i + 1)
        airfoils.append(df)

    return {
        'blade_data': blade_data,
        'settings': settings,
        'airfoil_shapes': shapes,
        'airfoils': dict(zip(valid_blafids, airfoils)),
        'valid_blafids': valid_blafids
    }

def calculate_optimal_strategy(operational_data):
    """Calculate optimal pitch and RPM strategy from operational data"""
    wind_speeds = operational_data[:, 0]
    pitches = operational_data[:, 1]
    rpms = operational_data[:, 2]
    
    pitch_interp = interp1d(wind_speeds, pitches, kind='linear', fill_value='extrapolate')
    rpm_interp = interp1d(wind_speeds, rpms, kind='linear', fill_value='extrapolate')
    
    return pitch_interp, rpm_interp, min(wind_speeds), max(wind_speeds)

def generate_wind_speed_range(min_speed, max_speed, step=0.1):
    """Generate wind speed range for optimal strategy"""
    return np.arange(min_speed, max_speed + step, step)

def perform_bem_calculations(inputs):
    """Perform baseline BEM calculations"""
    summary_results = []
    blade_data = inputs['blade_data']
    settings = inputs['settings']
    
    for _, row in settings.iterrows():
        V0 = row['WindSpeed']
        pitch = row['PitchAngle']
        omega = calc_omega(row['RotSpeed'])
        
        # Run BEM calculations for this condition
        condition_results = calculate_single_condition(
            V0, pitch, omega,
            blade_data['BlSpn'].values,
            blade_data['BlSpn'],
            blade_data['BlTwist'],
            blade_data['BlChord'],
            blade_data['BlAFID'],
            inputs['valid_blafids'],
            inputs['airfoils'],
            NUMBER_BLADES, R, RHO, MAX_ITER, TOLERANCE
        )
        summary_results.append(condition_results['summary'])
    
    return summary_results

def calculate_single_condition(V0, pitch, omega, r_s, BlSpn, BlTwist, BlChord,
                             BlAFID, valid_blafids, airfoils, num_blades,
                             radius, rho, max_iter, tolerance):
    """Calculate BEM solution for single condition"""
    elemental = defaultdict(list)
    dT_list, dM_list, r_used = [], [], []
    
    for r, af_id in zip(r_s, BlAFID):
        beta, chord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)
        df_polar = airfoils[valid_blafids.index(af_id)]

        bem_result = solve_blade_element(
            V0, omega, r, pitch, beta, chord, df_polar, num_blades,
            radius, rho, max_iter, tolerance, af_id
        )
        
        if bem_result is None:
            continue
            
        dT_list.append(num_blades * bem_result['p_n'])
        dM_list.append(num_blades * r * bem_result['p_t'])
        r_used.append(r)
    
    T, M, P_total, CP, CT = compute_totals_and_coefficients(
        dT_list, dM_list, omega, r_used, rho, radius, V0, RATED_POWER
    )
    
    return {
        'summary': {
            "V_0": V0, "T": T, "M": M, "P": P_total,
            "C_P": CP, "C_T": CT, "omega": omega, "pitch": pitch
        }
    }

def solve_blade_element(V0, omega, r, pitch, beta, chord, df_polar,
                       num_blades, radius, rho, max_iter, tolerance, af_id):
    """Core BEM solver for single element"""
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
        'p_t': p_t
    }

def run_bem_analysis():
    """Main analysis function"""
    start_time = time()
    inputs = load_input_data()
    
    # Baseline analysis
    baseline_results = perform_bem_calculations(inputs)
    baseline_df = pd.DataFrame(baseline_results).sort_values("V_0")
    
    # Optimal strategy
    operational_data = inputs['settings'][['WindSpeed', 'PitchAngle', 'RotSpeed']].values
    pitch_interp, rpm_interp, min_ws, max_ws = calculate_optimal_strategy(operational_data)
    wind_speed_range = generate_wind_speed_range(min_ws, max_ws)
    
    optimal_results = []
    for V0 in wind_speed_range:
        pitch = float(pitch_interp(V0))
        rpm = float(rpm_interp(V0))
        omega = calc_omega(rpm)
        
        condition_results = calculate_single_condition(
            V0, pitch, omega,
            inputs['blade_data']['BlSpn'].values,
            inputs['blade_data']['BlSpn'],
            inputs['blade_data']['BlTwist'],
            inputs['blade_data']['BlChord'],
            inputs['blade_data']['BlAFID'],
            inputs['valid_blafids'],
            inputs['airfoils'],
            NUMBER_BLADES, R, RHO, MAX_ITER, TOLERANCE
        )
        optimal_results.append(condition_results['summary'])
    
    optimal_df = pd.DataFrame(optimal_results).sort_values("V_0")
    
    # Save and plot results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    baseline_df.to_csv(output_dir / "baseline_performance.csv", index=False)
    optimal_df.to_csv(output_dir / "optimal_performance.csv", index=False)
    
    # Generate comparison plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(baseline_df["V_0"], baseline_df["P"]/1e6, 'b-', label='Baseline')
    plt.plot(optimal_df["V_0"], optimal_df["P"]/1e6, 'r-', label='Optimal')
    plt.xlabel('Wind speed [m/s]'); plt.ylabel('Power [MW]')
    plt.grid(); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(baseline_df["V_0"], baseline_df["T"]/1000, 'b-', label='Baseline')
    plt.plot(optimal_df["V_0"], optimal_df["T"]/1000, 'r-', label='Optimal')
    plt.xlabel('Wind speed [m/s]'); plt.ylabel('Thrust [kN]')
    plt.grid(); plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Execution time: {time() - start_time:.2f} s")

if __name__ == "__main__":
    run_bem_analysis()
