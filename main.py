import os
import numpy as np
import matplotlib.pyplot as plt

from src import (
    load_blade_geometry,
    load_airfoil_polars,
    load_operational_strategy,
    load_all_shapes,
    plot_airfoil_shapes,
    load_all_polars
)

# ---------- BASE PATH ----------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # directory where main.py is located
INPUT_DIR = os.path.join(BASE_PATH, "inputs", "IEA-15-240-RWT")

# ---------- FILE PATHS ----------
BLADE_GEOM_PATH = os.path.join(INPUT_DIR, "IEA-15-240-RWT_AeroDyn15_blade.dat")
OPS_STRATEGY_PATH = os.path.join(INPUT_DIR, "IEA_15MW_RWT_Onshore.opt")
POLAR_FOLDER = os.path.join(INPUT_DIR, "Airfoils")
SHAPE_FOLDER = os.path.join(INPUT_DIR, "Airfoils")
ROTOR_RADIUS = 120  # meters
RHO = 1.225  # kg/m^3 (air density)

# ---------- LOAD DATA ----------
blade_df = load_blade_geometry(BLADE_GEOM_PATH)
shape_data = load_all_shapes(SHAPE_FOLDER)
strategy_df = load_operational_strategy(OPS_STRATEGY_PATH)
all_polars = load_all_polars(POLAR_FOLDER)


# plot airfoil shapes
plot_airfoil_shapes(shape_data)







# # ----------------------------
# # Cl and Cd Distribution
# # ----------------------------
# print("Computing Cl/Cd distribution for α = 6°...")
# r, cl, cd = compute_cl_cd_distribution(blade_df, polar_dict, alpha_deg=6)

# plt.figure()
# plt.plot(r, cl, label='Cl')
# plt.plot(r, cd, label='Cd')
# plt.xlabel("Span position [m]")
# plt.ylabel("Coefficient")
# plt.title("Cl and Cd vs Blade Span at α = 6°")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # Induction Factors
# # ----------------------------
# print("Computing induction factors for V₀ = 8 m/s...")
# r, a, a_prime = compute_induction_factors(blade_df, polar_dict, V0=8, pitch_deg=2, omega=1.0)

# plt.figure()
# plt.plot(r, a, label='Axial induction factor (a)')
# plt.plot(r, a_prime, label="Tangential induction factor (a')")
# plt.xlabel("Span position [m]")
# plt.ylabel("Induction Factor")
# plt.title("Induction Factors vs Blade Span")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # T, M, P at One Operating Point
# # ----------------------------
# print("Computing T, M, P at V₀ = 10 m/s...")
# T, M, P = compute_thrust_torque_power(blade_df, polar_dict, V0=10, pitch_deg=3, omega=1.2)
# print(f"Thrust: {T:.1f} N, Torque: {M:.1f} Nm, Power: {P/1e6:.2f} MW")

# # ----------------------------
# # Optimal Operating Point
# # ----------------------------
# print("Interpolating optimal pitch and omega for V₀ = 9 m/s...")
# pitch, omega = get_optimal_operating_point(strategy_df, V0_target=9)
# print(f"Optimal pitch: {pitch:.2f}°, omega: {omega:.3f} rad/s")

# # ----------------------------
# # Power and Thrust Curves
# # ----------------------------
# print("Plotting power and thrust curves...")
# plot_power_thrust_curves(blade_df, polar_dict, strategy_df)