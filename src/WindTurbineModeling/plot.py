import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO

# TODO - Add docstrings to all functions


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

import matplotlib.pyplot as plt
import numpy as np

def plot_power_thrust_curves(wind_speeds, powers, thrusts):
    """
    Plot Power and Thrust curves vs. Wind Speed
    """
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Wind Speed (m/s)")
    ax1.set_ylabel("Power (MW)", color="tab:blue")
    ax1.plot(wind_speeds, np.array(powers)/1e6, label="Power", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Thrust (kN)", color="tab:red")
    ax2.plot(wind_speeds, np.array(thrusts)/1e3, label="Thrust", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Power and Thrust Curves")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

def plot_wind_turbine(R=10, r=3, dr=1, r_hub=0.6, tower_height=25, num_blades=3):
    """
    Plot a stylized 2D wind turbine with labeled radial dimensions.

    Parameters:
    -----------
    R : float
        Outer radius (blade length) from hub center.
    r : float
        Radius of inner dashed circle (e.g., start of analysis area).
    dr : float
        Width of the differential ring (drawn between r and R-dr).
    r_hub : float
        Radius of the hub (central circle).
    tower_height : float
        Height of the turbine tower.
    num_blades : int
        Number of turbine blades.

    The diagram includes:
    - A tower and circular hub
    - Tapered wind turbine blades
    - Dashed radial overlays for r, R-dr, and R
    - Labeled arrows for r, R, and dr

    Created by ChatGPT (OpenAI), April 2025
    """

    # Derived parameters
    hub_height = tower_height + r_hub
    theta = np.linspace(0, 2 * np.pi, num_blades, endpoint=False)

    # Blade drawing function (local)
    def draw_offset_blade(ax, angle, r_hub, R, y_offset):
        blade_length = R - r_hub
        x = np.array([0, 0.2, 0.5, 0.8, 1.0]) * blade_length + r_hub
        y = np.array([0, 0.3, 0.25, 0.15, 0])  # Tapered half-width
        x_full = np.concatenate([x, x[::-1]])
        y_full = np.concatenate([y, -y[::-1]])
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        coords = np.vstack([x_full, y_full]).T @ rot.T
        coords[:, 1] += y_offset
        ax.fill(coords[:, 0], coords[:, 1], 'white', edgecolor='black', linewidth=1)

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw tower
    ax.plot([0, 0], [0, tower_height], color='black', linewidth=4)

    # Draw hub
    hub = plt.Circle((0, hub_height), r_hub, color='white', ec='black', linewidth=1.5)
    ax.add_artist(hub)

    # Draw dashed circles
    r_ring = plt.Circle((0, hub_height), r, color='black', linestyle='--', fill=False, linewidth=1)
    inner_ring = plt.Circle((0, hub_height), R - dr, color='black', linestyle='--', fill=False, linewidth=1)
    outer_circle = plt.Circle((0, hub_height), R, color='black', linestyle='--', fill=False, linewidth=1)
    ax.add_artist(r_ring)
    ax.add_artist(inner_ring)
    ax.add_artist(outer_circle)

    # Draw blades
    for angle in theta:
        draw_offset_blade(ax, angle, r_hub, R, hub_height)

    # Arrows and labels
    arrowprops = dict(arrowstyle='->', linewidth=1.2)
    ax.annotate('', xy=(r, hub_height), xytext=(0, hub_height), arrowprops=arrowprops)
    ax.text(r / 2 - 0.3, hub_height + 0.5, 'r', fontsize=14)

    ax.annotate('', xy=(R, hub_height), xytext=(0, hub_height), arrowprops=arrowprops)
    ax.text(R / 2 - 0.5, hub_height + 0.8, 'R', fontsize=14)

    dr_middle = (r + dr) / 2
    ax.annotate('', xy=(R - dr, hub_height), xytext=(r, hub_height),
                arrowprops=dict(arrowstyle='<->', linewidth=1.2))
    ax.text(dr_middle - 0.3, hub_height + 0.4, 'dr', fontsize=14)

    # Limits
    ax.set_xlim(-R - 2, R + 2)
    ax.set_ylim(-5, tower_height + R + 2)
    plt.show()


# Call the function to test it
#plot_wind_turbine()


def plot_lift_drag_vs_span(r_s, alpha_list, Cl_list, Cd_list):
    """
    Plot Cl and Cd as functions of spanwise position and angle of attack.

    Parameters:
    -----------
    r_s : array-like
        Spanwise positions [m]
    alpha_list : list of float
        Mean angle of attack [deg] at each r
    Cl_list : list of float
        Mean lift coefficient at each r
    Cd_list : list of float
        Mean drag coefficient at each r
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_title("Lift and Drag Coefficients vs Span Position")
    ax1.plot(r_s, Cl_list, label="Cl", color="blue")
    ax1.plot(r_s, Cd_list, label="Cd", color="red")
    ax1.set_xlabel("Span position r [m]")
    ax1.set_ylabel("Coefficient Value")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Add α as a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(r_s, alpha_list, '--', color='gray', label="Alpha [deg]")
    ax2.set_ylabel("Angle of Attack α [deg]")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_induction_factors_vs_span(r_s, a_list, a_prime_list):
    """
    Plot axial (a) and tangential (a') induction factors vs spanwise position.

    Parameters:
    -----------
    r_s : array-like
        Spanwise positions [m]
    a_list : list of float
        Axial induction factor values at each r
    a_prime_list : list of float
        Tangential induction factor values at each r
    """
    plt.figure(figsize=(10, 5))
    plt.plot(r_s, a_list, label="a (axial)", color='green')
    plt.plot(r_s, a_prime_list, label="a' (tangential)", color='orange')
    plt.title("Induction Factors vs Span Position")
    plt.xlabel("Span position r [m]")
    plt.ylabel("Induction Factor [-]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_power_curve(V_arr, P_arr, RATED_POWER):
    """
    Plot the power curve of the wind turbine.

    Parameters:
    V_arr : array-like
        Wind speeds [m/s].
    P_arr : array-like
        Power output [MW].
    rated_power : float
        Rated power of the turbine [MW].
    """
    plt.figure(figsize=(10, 5))
    plt.plot(V_arr, P_arr, label="Power Curve", color='blue')
    plt.axhline(y=RATED_POWER / 1e6, color='r', linestyle='--', label='Rated Power')
    plt.title("Wind Turbine Power Curve(BEM)")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power Output (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cp_curve(V_arr, Cp_arr):
    """
    Plot the power coefficient curve of the wind turbine.

    Parameters:
    V_arr : array-like
        Wind speeds [m/s].
    Cp_arr : array-like
        Power coefficients [-].
    """
    plt.figure(figsize=(10, 5))
    plt.plot(V_arr, Cp_arr, label="Cp Curve", color='green')
    plt.title("Wind Turbine Power Coefficient Curve")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power Coefficient (Cp)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_thrust_curve(V_arr, T_arr):
    """
    Plot the thrust curve of the wind turbine.

    Parameters:
    V_arr : array-like
        Wind speeds [m/s].
    T_arr : array-like
        Thrust forces [kN].
    """
    plt.figure(figsize=(10, 5))
    plt.plot(V_arr, T_arr / 1e3, label="Thrust Curve", color='red')
    plt.title("Wind Turbine Thrust Curve(BEM)")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Thrust Force (kN)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ct_curve(V_arr, Ct_arr):
    """
    Plot the thrust coefficient curve of the wind turbine.

    Parameters:
    V_arr : array-like
        Wind speeds [m/s].
    Ct_arr : array-like
        Thrust coefficients [-].
    """
    plt.figure(figsize=(10, 5))
    plt.plot(V_arr, Ct_arr, label="Ct Curve", color='purple')
    plt.title("Wind Turbine Thrust Coefficient Curve")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Thrust Coefficient (Ct)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
 
def plot_power_curve_opt(wind_speeds, power_output, rated_power=None):
    plt.figure(figsize=(8, 5))
    plt.plot(wind_speeds, power_output, 'b-', linewidth=2, label='Power Output')
    
    if rated_power is not None:
        plt.axhline(y=rated_power, color='r', linestyle='--', label='Rated Power')
    
    plt.title('Power Curve (Optimal Control Strategy)')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power [MW]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_thrust_curve_opt(wind_speeds, thrust_values):
    plt.figure(figsize=(8, 5))
    plt.plot(wind_speeds, thrust_values, 'g-', linewidth=2, label='Thrust')
    
    plt.title('Thrust Curve (Optimal Control Strategy)')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Thrust [N]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()