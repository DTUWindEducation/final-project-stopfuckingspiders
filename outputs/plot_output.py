import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Import your existing modules
from config import *
from WindTurbineModeling.read import *
from WindTurbineModeling.load import *
from WindTurbineModeling.equations import *
from WindTurbineModeling.plot import *
def run_bem_analysis():
    """Main function to run the complete BEM analysis"""
    start_time = time()
    
    # Load all input data
    inputs = load_input_data()
    
    # Run BEM calculations
    summary_results, elemental_data = perform_bem_calculations(inputs)
    
    # Process and save results
    save_results(summary_results, elemental_data, inputs['airfoils'])
    
    # Generate plots
    generate_plots(summary_results, inputs['airfoil_shapes'])
    
    print(f"Total execution time: {time() - start_time:.2f} seconds")

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

def perform_bem_calculations(inputs):
    """Perform BEM calculations for all operational conditions"""
    summary_results = []
    elemental_data = defaultdict(list)
    
    blade_data = inputs['blade_data']
    settings = inputs['settings']
    valid_blafids = inputs['valid_blafids']
    airfoils = inputs['airfoils']
    
    r_s = blade_data['BlSpn'].values
    BlSpn = blade_data['BlSpn']
    BlTwist = blade_data['BlTwist']
    BlChord = blade_data['BlChord']
    BlAFID = blade_data['BlAFID']

    for _, row in settings.iterrows():
        V0 = row['WindSpeed']
        pitch = row['PitchAngle']
        omega = calc_omega(row['RotSpeed'])
        
        # Run BEM for this operational condition
        condition_results = calculate_single_condition(
            V0, pitch, omega, r_s, BlSpn, BlTwist, BlChord, BlAFID,
            valid_blafids, airfoils, NUMBER_BLADES, R, RHO,
            MAX_ITER, TOLERANCE
        )
        
        # Store results
        summary_results.append(condition_results['summary'])
        for key, values in condition_results['elemental'].items():
            elemental_data[key].extend(values)
    
    return summary_results, elemental_data

def calculate_single_condition(V0, pitch, omega, r_s, BlSpn, BlTwist, BlChord,
                             BlAFID, valid_blafids, airfoils, num_blades,
                             radius, rho, max_iter, tolerance):
    """Calculate BEM solution for a single wind speed/pitch combination"""
    elemental = defaultdict(list)
    dT_list, dM_list, r_used = [], [], []
    
    for r, af_id in zip(r_s, BlAFID):
        # Get local blade geometry
        beta, chord = interpolate_blade_geometry(r, BlSpn, BlTwist, BlChord)
        df_polar = airfoils[valid_blafids.index(af_id)]

        # Run BEM iteration for this blade element
        bem_result = solve_blade_element(
            V0, omega, r, pitch, beta, chord, df_polar, num_blades,
            radius, rho, max_iter, tolerance, af_id
        )
        
        if bem_result is None:
            continue  # Skip invalid solutions
            
        # Store elemental data
        for key, value in bem_result['elemental'].items():
            elemental[key].append(value)
        
        # Store for summary calculations
        dT_list.append(num_blades * bem_result['p_n'])
        dM_list.append(num_blades * r * bem_result['p_t'])
        r_used.append(r)
    
    # Calculate summary values
    T, M, P_total, CP, CT = compute_totals_and_coefficients(
        dT_list, dM_list, omega, r_used, rho, radius, V0, RATED_POWER
    )
    
    return {
        'summary': {
            "V_0": V0,
            "T": T,
            "M": M,
            "P": P_total,
            "C_P": CP,
            "C_T": CT,
            "omega": omega,
            "pitch": pitch
        },
        'elemental': elemental
    }

def solve_blade_element(V0, omega, r, pitch, beta, chord, df_polar,
                       num_blades, radius, rho, max_iter, tolerance, af_id):
    """Solve BEM equations for a single blade element"""
    a, a_prime = 0.0, 0.0
    
    for _ in range(max_iter):
        # Flow angle and AoA
        phi = calc_flow_angle(V0, omega, r, a, a_prime)
        alpha = calc_local_angle_of_attack(phi, pitch, beta)
        
        # Aerodynamic coefficients
        Cl, Cd = calc_local_lift_drag_force(alpha, df_polar)
        Cn, Ct = calc_normal_tangential_constants(phi, Cl, Cd)
        
        # Local solidity and tip loss
        sigma = calc_local_solidity(num_blades, chord, r)
        F = calc_prandtl_tip_loss(num_blades, radius, r, phi)
        
        # Update induction factors
        a_new, a_prime_new = update_induction_factors(phi, sigma, Cn, Ct, F, a)
        
        # Check convergence
        if abs(a_new - a) < tolerance and abs(a_prime_new - a_prime) < tolerance:
            break
            
        a, a_prime = a_new, a_prime_new
    
    # Compute final forces
    v_rel, p_n, p_t = calc_relative_velocity_and_forces(
        V0, omega, r, a, a_prime, rho, chord, Cn, Ct
    )
    
    # Validate solution
    if np.isnan(Cl) or np.isnan(Cd) or np.isnan(p_t) or Cl == 0 or p_t < 0:
        return None
    
    return {
        'p_n': p_n,
        'p_t': p_t,
        'elemental': {
            'V0': V0,
            'r': r,
            'alpha': alpha,
            'Cl': Cl,
            'Cd': Cd,
            'a': a,
            'a_prime': a_prime,
            'phi': phi,
            'F': F,
            'p_n': p_n,
            'p_t': p_t,
            'sigma': sigma,
            'v_rel': v_rel,
            'beta': beta,
            'chord': chord,
            'af_id': af_id
        }
    }

def save_results(summary_results, elemental_data, airfoil_data):
    """Save all results to CSV files"""
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Save summary performance data
    summary_df = pd.DataFrame(summary_results).sort_values("V_0")
    summary_df.to_csv(output_dir / "performance_summary.csv", index=False)
    
    # 2. Save elemental data (single CSV)
    elemental_df = pd.DataFrame(elemental_data)
    elemental_df.to_csv(output_dir / "elemental_data.csv", index=False)
    
    # 3. Optional: Save by wind speed
    for v0 in elemental_df['V0'].unique():
        elemental_df[elemental_df['V0'] == v0].to_csv(
            output_dir / f"elemental_data_{v0}mps.csv",
            index=False
        )
    
    # 4. Save airfoil data
    airfoil_dir = output_dir / "airfoils"
    airfoil_dir.mkdir(exist_ok=True)
    for af_id, df in airfoil_data.items():
        df.to_csv(airfoil_dir / f"airfoil_{af_id}.csv", index=False)

def generate_plots(summary_results, airfoil_shapes):
    """Generate all standard plots"""
    summary_df = pd.DataFrame(summary_results).sort_values("V_0")
    V_arr = summary_df["V_0"].values
    P_arr = summary_df["P"].values/1e6
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

if __name__ == "__main__":
    run_bem_analysis()