import matplotlib.pyplot as plt
import numpy as np

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
plot_wind_turbine()
