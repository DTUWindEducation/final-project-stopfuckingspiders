import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO

def load_blade_geometry(filepath):
    """
    Load blade geometry from AeroDyn15 file:
    Extracts BlSpn, BlTwist, BlChord, and BlAFID.
    """
    df = pd.read_csv(
        filepath,
        sep='\s+',
        skiprows=6,  # skip header, units, and metadata
        header=None,
        names=[
            'BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist',
            'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt'
        ]
    )

    # Select and cast relevant columns
    blade_geom = df[['BlSpn', 'BlTwist', 'BlChord', 'BlAFID']]
    blade_geom.loc[:, 'BlAFID'] = blade_geom['BlAFID'].astype(int)

    return blade_geom

def load_operational_strategy(filepath):
    """
    Loads the operational strategy data, skipping the header line.
    """
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=1,  # Skip the malformed "header" row
        header=None,
        names=['WindSpeed', 'PitchAngle', 'RotSpeed', 'AeroPower', 'AeroThrust']
    )
    return df

def load_airfoil_shape(filepath):
    """
    Load normalized airfoil shape coordinates from the airfoil shape file.
    """
    data = np.loadtxt(filepath, skiprows=8)
    return pd.DataFrame(data, columns=["x/c", "y/c"])

def load_all_shapes(shape_dir):
    shapes = []
    for i in range(50):
        idx = f"{i:02d}"
        path = os.path.join(shape_dir, f"IEA-15-240-RWT_AF{idx}_Coords.txt")
        if os.path.exists(path):
            df = load_airfoil_shape(path)
            shapes.append(df)
    return shapes

import pandas as pd

def load_airfoil_polars(filepath, skiprows=55):
    """
    Load aerodynamic polar data from an AirfoilInfo file.
    Handles files with variable number of columns.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep=r'\s+',
            skiprows=skiprows,
            header=None,
            engine='python',
            on_bad_lines='skip'  # skip malformed rows
        )

        if df.shape[1] < 3:
            raise ValueError(f"[Error] Not enough columns in {filepath}.")

        df = df.iloc[:, :3]
        df.columns = ['Alpha', 'Cl', 'Cd']
        return df

    except Exception as e:
        print(f"[Warning] Failed to load {filepath}: {e}")
        return None

def load_all_polars(folder):
    polar_dict = {}
    for i in range(50):
        airfoil_id = i + 1
        filename = f"IEA-15-240-RWT_AeroDyn15_Polar_{i:02d}.dat"
        filepath = os.path.join(folder, filename)

        # Use different skiprows if needed (based on inspection)
        skiprows = 21 if i < 5 else 55
        df = load_airfoil_polars(filepath, skiprows=skiprows)
        if df is not None:
            polar_dict[airfoil_id] = df

    return polar_dict






# def get_cl_cd(alpha, polar_df):
#     alpha = np.clip(alpha, polar_df["Alpha"].min(), polar_df["Alpha"].max())
#     alpha_col = pd.to_numeric(polar_df["Alpha"], errors="coerce")
#     cl_col = pd.to_numeric(polar_df["Cl"], errors="coerce")
#     cd_col = pd.to_numeric(polar_df["Cd"], errors="coerce")
#     valid = ~(alpha_col.isna() | cl_col.isna() | cd_col.isna())
#     alpha_col, cl_col, cd_col = alpha_col[valid], cl_col[valid], cd_col[valid]
#     cl = np.interp(alpha, alpha_col, cl_col)
#     cd = np.interp(alpha, alpha_col, cd_col)
#     return cl, cd

# def load_all_polars(polar_dir):
#     polars = {}
#     for i in range(50):
#         idx = f"{i:02d}"
#         path = os.path.join(polar_dir, f"IEA-15-240-RWT_AeroDyn15_Polar_{idx}.dat")
#         polars[i] = load_polar_data(path)
#     return polars

# def load_all_shapes(shape_dir):
#     shapes = []
#     for i in range(50):
#         idx = f"{i:02d}"
#         path = os.path.join(shape_dir, f"IEA-15-240-RWT_AF{idx}_Coords.txt")
#         if os.path.exists(path):
#             df = load_airfoil_shape(path)
#             shapes.append(df)
#     return shapes

# # ----------------------------------
# # BEM Solver
# # ----------------------------------
# def solve_bem(blade_df, polar_dict, V0, pitch, omega, rho=1.225, B=3, R=120.0):
#     r = blade_df["BlSpn"].values
#     dr = np.gradient(r)
#     T, M = 0, 0

#     for i in range(len(blade_df)):
#         ri = r[i]
#         if ri == 0: continue
#         c = blade_df["BlChord"].iloc[i]
#         beta = blade_df["BlTwist"].iloc[i]
#         afid = int(blade_df["BlAFID"].iloc[i])
#         polar_df = polar_dict.get(afid)
#         if polar_df is None: continue

#         sigma = B * c / (2 * np.pi * ri)
#         a, a_prime = 0.0, 0.0

#         for _ in range(100):
#             phi = np.arctan2((1 - a) * V0, (1 + a_prime) * omega * ri)
#             alpha = np.degrees(phi) - (pitch + beta)
#             cl, cd = get_cl_cd(alpha, polar_df)
#             cn = cl * np.cos(phi) + cd * np.sin(phi)
#             ct = cl * np.sin(phi) - cd * np.cos(phi)
#             denom_a = 4 * np.sin(phi)**2 / (sigma * cn) + 1
#             denom_ap = 4 * np.sin(phi) * np.cos(phi) / (sigma * ct) - 1
#             if denom_a == 0 or denom_ap == 0: break
#             anew = 1 / denom_a
#             aprime_new = 1 / denom_ap
#             if np.abs(anew - a) < 1e-5 and np.abs(aprime_new - a_prime) < 1e-5: break
#             a, a_prime = anew, aprime_new
#         if i % 10 == 0:
#             print(f"[Section {i}] r={ri:.1f}, α={alpha:.2f}, Cl={cl:.3f}, Cd={cd:.3f}, Cn={cn:.3f}, Ct={ct:.3f}")
#         dT = 4 * np.pi * ri * rho * V0**2 * a * (1 - a) * dr[i]
#         dM = 4 * np.pi * ri**3 * rho * V0 * omega * a_prime * (1 - a) * dr[i]

#         T += dT
#         M += dM

#     P = M * omega
#     return T, M, P

# # ----------------------------------
# # Performance
# # ----------------------------------
# def compute_power_thrust_curves(blade_df, polar_dict, strategy_df, rho=1.225, R=120.0):
#     wind_speeds, thrusts, powers = [], [], []
#     for _, row in strategy_df.iterrows():
#         V0 = row["wind_speed"]
#         pitch = row["pitch"]
#         omega = row["rot_speed"] * 2 * np.pi / 60
#         T, M, P = solve_bem(blade_df, polar_dict, V0, pitch, omega, rho=rho, R=R)
#         wind_speeds.append(V0)
#         thrusts.append(T)
#         powers.append(P)
#     return np.array(wind_speeds), np.array(thrusts), np.array(powers)

# # ----------------------------------
# # Plotting
# # ----------------------------------
def plot_airfoil_shapes(airfoil_data_list, labels=None):
    plt.figure(figsize=(12, 6))
    for i, df in enumerate(airfoil_data_list):
        x, y = df["x/c"], df["y/c"]
        label = labels[i] if labels else f"AF{i:02d}"
        plt.plot(x, y, label=label, linewidth=1, alpha=0.7)
    plt.title("Airfoil Shapes")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(ncol=3, fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# def plot_power_thrust_curves(blade_df, polar_dict, strategy_df, rho=1.225, R=120):
#     V0s, thrusts, powers = compute_power_thrust_curves(blade_df, polar_dict, strategy_df, rho=rho, R=R)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(V0s, powers / 1e6, 'r-')
#     plt.xlabel("Wind Speed [m/s]")
#     plt.ylabel("Power [MW]")
#     plt.title("Power Curve")
#     plt.grid(True)

#     plt.subplot(1, 2, 2)
#     plt.plot(V0s, thrusts / 1e3, 'b-')
#     plt.xlabel("Wind Speed [m/s]")
#     plt.ylabel("Thrust [kN]")
#     plt.title("Thrust Curve")
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

# # ----------------------------------
# # Utilities
# # ----------------------------------
# def tip_speed_ratio(omega, R, V0):
#     return omega * R / V0

# def rotor_area(R):
#     return np.pi * R ** 2

# # ----------------------------------
# # Additional Assignment Functions
# # ----------------------------------
# def compute_cl_cd_distribution(blade_df, polar_dict, alpha_deg):
#     span = blade_df["BlSpn"]
#     cl_values, cd_values = [], []
#     for i, row in blade_df.iterrows():
#         afid = int(row["BlAFID"])
#         polar = polar_dict[afid]
#         cl, cd = get_cl_cd(alpha_deg, polar)
#         cl_values.append(cl)
#         cd_values.append(cd)
#     return np.array(span), np.array(cl_values), np.array(cd_values)

# def compute_induction_factors(blade_df, polar_dict, V0, pitch_deg, omega, rho=1.225, B=3):
#     a_list, a_prime_list, span_list = [], [], []
#     for _, row in blade_df.iterrows():
#         r = row["BlSpn"]
#         if r == 0: continue
#         c = row["BlChord"]
#         beta = row["BlTwist"]
#         afid = int(row["BlAFID"])
#         polar = polar_dict.get(afid)
#         if polar is None: continue
#         sigma = B * c / (2 * np.pi * r)
#         a, a_prime = 0.0, 0.0
#         for _ in range(100):
#             phi = np.arctan2((1 - a) * V0, (1 + a_prime) * omega * r)
#             alpha = np.degrees(phi) - (pitch_deg + beta)
#             cl, cd = get_cl_cd(alpha, polar)
#             cn = cl * np.cos(phi) + cd * np.sin(phi)
#             ct = cl * np.sin(phi) - cd * np.cos(phi)
#             a_new = 1 / (4 * np.sin(phi)**2 / (sigma * cn) + 1)
#             a_prime_new = 1 / (4 * np.sin(phi) * np.cos(phi) / (sigma * ct) - 1)
#             if np.abs(a_new - a) < 1e-5 and np.abs(a_prime_new - a_prime) < 1e-5: break
#             a, a_prime = a_new, a_prime_new
#         a_list.append(a)
#         a_prime_list.append(a_prime)
#         span_list.append(r)
#     return np.array(span_list), np.array(a_list), np.array(a_prime_list)

# def compute_thrust_torque_power(blade_df, polar_dict, V0, pitch_deg, omega, rho=1.225, R=120):
#     return solve_bem(blade_df, polar_dict, V0, pitch_deg, omega, rho=rho, R=R)

# def get_optimal_operating_point(strategy_df, V0_target):
#     df = strategy_df.sort_values("wind_speed")
#     pitch = np.interp(V0_target, df["wind_speed"], df["pitch"])
#     omega_rpm = np.interp(V0_target, df["wind_speed"], df["rot_speed"])
#     omega_rad = omega_rpm * 2 * np.pi / 60
#     return pitch, omega_rad
