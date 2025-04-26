import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO

def plot_airfoil_shapes(airfoil_data_list, labels=None):
    """
    Plot a collection of airfoil shapes (x/c vs y/c) from a list of dataframes.

    Parameters:
    ----------
    airfoil_data_list : list of pd.DataFrame
        Each dataframe must contain 'x/c' and 'y/c' columns representing airfoil geometry.

    labels : list of str, optional
        Optional custom labels for each airfoil in the legend. If None, default names like AF01, AF02, ... are used.
    """
    plt.figure(figsize=(12, 6))
    for i, df in enumerate(airfoil_data_list):
        x, y = df["x/c"], df["y/c"]
        label = labels[i] if labels else f"AF{i + 1:02d}"
        plt.plot(x, y, label=label, linewidth=1, alpha=0.7)

    plt.title("Airfoil Profiles")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(ncol=3, fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_wind_turbine(R=10, r=3, dr=1, r_hub=0.6, tower_height=25, num_blades=3):
    """
    Draw a stylized top-view schematic of a wind turbine with blade geometry and labeled radii.

    Parameters:
    ----------
    R : float
        Total blade length from hub center [m].
    r : float
        Start radius for aerodynamic calculations (inner dashed arc) [m].
    dr : float
        Differential ring width shown on the rotor [m].
    r_hub : float
        Radius of the central hub circle [m].
    tower_height : float
        Vertical offset of the rotor above ground [m].
    num_blades : int
        Number of blades to draw.
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    hub_height = tower_height + r_hub
    blade_angles = np.linspace(0, 2 * np.pi, num_blades, endpoint=False)

    def draw_blade(ax, angle):
        blade_length = R - r_hub
        x = np.array([0, 0.2, 0.5, 0.8, 1.0]) * blade_length + r_hub
        y = np.array([0, 0.3, 0.25, 0.15, 0])
        x_full = np.concatenate([x, x[::-1]])
        y_full = np.concatenate([y, -y[::-1]])
        coords = np.vstack([x_full, y_full]).T @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        coords[:, 1] += hub_height
        ax.fill(coords[:, 0], coords[:, 1], 'white', edgecolor='black', linewidth=1)

    # Tower and hub
    ax.plot([0, 0], [0, tower_height], color='black', linewidth=4)
    ax.add_artist(plt.Circle((0, hub_height), r_hub, color='white', ec='black', linewidth=1.5))

    # Circles and blades
    for radius in [r, R - dr, R]:
        ax.add_artist(plt.Circle((0, hub_height), radius, color='black', linestyle='--', fill=False, linewidth=1))

    for angle in blade_angles:
        draw_blade(ax, angle)

    # Dimension labels
    ax.annotate('', xy=(r, hub_height), xytext=(0, hub_height), arrowprops=dict(arrowstyle='->', linewidth=1.2))
    ax.text(r / 2 - 0.3, hub_height + 0.5, 'r', fontsize=14)

    ax.annotate('', xy=(R, hub_height), xytext=(0, hub_height), arrowprops=dict(arrowstyle='->', linewidth=1.2))
    ax.text(R / 2 - 0.5, hub_height + 0.8, 'R', fontsize=14)

    ax.annotate('', xy=(R - dr, hub_height), xytext=(r, hub_height), arrowprops=dict(arrowstyle='<->', linewidth=1.2))
    ax.text((r + R - dr) / 2 - 0.3, hub_height + 0.4, 'dr', fontsize=14)

    ax.set_xlim(-R - 2, R + 2)
    ax.set_ylim(-5, tower_height + R + 2)
    plt.show()

def plot_power_curve(V_arr, P_arr, rated_power_watts):
    """
    Plot power curve (wind speed vs. power) including rated power reference.

    Parameters:
    ----------
    V_arr : ndarray
        Array of wind speeds [m/s].
    P_arr : ndarray
        Corresponding power values [MW].
    rated_power_watts : float
        Rated power in watts to mark on the curve.
    """
    plt.figure()
    plt.plot(V_arr, P_arr, label='Power [MW]', marker='o', linewidth=2)
    plt.axhline(rated_power_watts / 1e6, color='r', linestyle='--', label='Rated Power')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power [MW]')
    plt.title('Power Curve')
    plt.grid(True)
    plt.legend()

def plot_cp_curve(V_arr, Cp_arr):
    """
    Plot the power coefficient curve (Cp vs. wind speed) with Betz limit.

    Parameters:
    ----------
    V_arr : ndarray
        Array of wind speeds [m/s].
    Cp_arr : ndarray
        Corresponding power coefficient values.
    """
    plt.figure()
    plt.plot(V_arr, Cp_arr, label='Cp', marker='o', linewidth=2)
    plt.axhline(16 / 27, color='orange', linestyle='--', label='Betz Limit')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Cp [-]')
    plt.title('Power Coefficient Curve')
    plt.grid(True)
    plt.legend()

def plot_thrust_curve(V_arr, T_arr):
    """
    Plot the thrust curve (wind speed vs. thrust force).

    Parameters:
    ----------
    V_arr : ndarray
        Array of wind speeds [m/s].
    T_arr : ndarray
        Corresponding thrust values [N].
    """
    plt.figure()
    plt.plot(V_arr, T_arr, label='Thrust [N]', marker='o', linewidth=2)
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Thrust [N]')
    plt.title('Thrust Curve')
    plt.grid(True)
    plt.legend()

