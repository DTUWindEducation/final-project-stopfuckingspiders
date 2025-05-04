import numpy as np
import matplotlib.pyplot as plt

def plot_airfoil_shapes(airfoil_data_list, labels=None):
    """
    Plots the shapes of airfoils based on provided data.

    Parameters:
        airfoil_data_list (list of pandas.DataFrame): A list of dataframes,
        each containing airfoil coordinates with columns "x/c" and "y/c".
        labels (list of str, optional): A list of labels for each airfoil.

    Returns:
        None: Displays the plot of airfoil shapes.
    """
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
    plt.legend(ncol=3, fontsize='x-small', bbox_to_anchor=(1.05, 1),
               loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_wind_turbine(R=10, r=3, dr=1, r_hub=0.6, tower_height=25,
                      num_blades=3):
    """
    Plots a schematic representation of a wind turbine.
    Parameters:
        R (float): Radius of the rotor (outermost blade length). Default is 10.
        r (float): Radius of the inner dashed circle. Default is 3.
        dr (float): Thickness of the rotor region (difference between R and
        the middle dashed circle). Default is 1.
        r_hub (float): Radius of the hub. Default is 0.6.
        tower_height (float): Height of the wind turbine tower. Default is 25.
        num_blades (int): Number of blades on the wind turbine. Default is 3.
    Returns:
        None: Displays the wind turbine plot.
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw tower
    ax.plot([0, 0], [0, tower_height], color='black', linewidth=4)

    # Draw hub
    hub_height = tower_height + r_hub
    hub = plt.Circle((0, hub_height), r_hub, color='white', ec='black',
                     linewidth=1.5)
    ax.add_artist(hub)

    # Draw blades
    angles = np.linspace(0, 2*np.pi, num_blades, endpoint=False)
    for angle in angles:
        blade_length = R - r_hub
        x = np.array([0, 0.2, 0.5, 0.8, 1.0]) * blade_length + r_hub
        y = np.array([0, 0.3, 0.25, 0.15, 0])
        coords = np.vstack([np.concatenate([x, x[::-1]]),
                            np.concatenate([y, -y[::-1]])]).T
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        coords = coords @ rot.T
        coords[:,1] += hub_height
        ax.fill(coords[:,0], coords[:,1], 'white',
                edgecolor='black', linewidth=1)

    # Dashed circles
    for radius in [r, R-dr, R]:
        circle = plt.Circle((0, hub_height), radius, color='black',
                            linestyle='--', fill=False, linewidth=1)
        ax.add_artist(circle)

    ax.set_xlim(-R-2, R+2)
    ax.set_ylim(-5, tower_height + R + 2)
    ax.set_title("Wind Turbine Schematic", pad=20, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_power_curve(wind_speeds, powers, rated_power=None):
    """
    Plots the power curve of a wind turbine.

    Parameters:
        wind_speeds (list or array-like): Wind speeds in m/s.
        powers (list or array-like): Corresponding power outputs in MW.
        rated_power (float, optional): Rated power of the turbine in watts.
        Defaults to None.

    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=(10,6))
    plt.plot(wind_speeds, powers, 'b-', label='Power Output')
    if rated_power:
        plt.axhline(rated_power / 1e6, color='r', linestyle='--',
                    label='Rated Power')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Power (MW)')
    plt.title('Wind Turbine Power Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_thrust_curve(wind_speeds, thrusts):
    """
    Plots the thrust curve of a wind turbine.

    Parameters:
    wind_speeds (list or array-like): Wind speeds in meters per second (m/s).
    thrusts (list or array-like): Corresponding thrust values in Newtons (N).

    Returns:
    None
    """
    plt.figure(figsize=(10,6))
    plt.plot(wind_speeds, np.array(thrusts)/1e3, 'g-', label='Thrust Curve')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Thrust (kN)')
    plt.title('Wind Turbine Thrust Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cp_curve(wind_speeds, cp_values):
    """
    Plots the power coefficient (Cp) curve against wind speeds.

    Parameters:
    wind_speeds (list or array-like): Wind speeds in meters per second.
    cp_values (list or array-like): Corresponding power coefficient values.

    Returns:
    None
    """
    plt.figure(figsize=(10,6))
    plt.plot(wind_speeds, cp_values, 'm-', label='Cp Curve')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Power Coefficient (Cp)')
    plt.title('Power Coefficient Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ct_curve(wind_speeds, ct_values):
    """
    Plots the thrust coefficient (Ct) curve against wind speeds.

    Parameters:
    wind_speeds (list or array-like): Wind speeds in meters per second.
    ct_values (list or array-like): Corresponding thrust coefficient values.

    Returns:
    None
    """
    plt.figure(figsize=(10,6))
    plt.plot(wind_speeds, ct_values, 'c-', label='Ct Curve')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Thrust Coefficient (Ct)')
    plt.title('Thrust Coefficient Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_induction_vs_span(elemental_data):
    """
    Plots axial (a) and tangential (a') induction factors against span (r) in
    two subplots, grouped by wind speed (V0).
    Parameters:
        elemental_data (dict): A dictionary containing the following keys:
            - 'r' (array-like): Span positions [m].
            - 'V0' (array-like): Wind speeds [m/s].
            - 'a' (array-like): Axial induction factors.
            - 'a_prime' (array-like): Tangential induction factors.
    Returns:
        None: Displays the plot.
    """
    r = np.array(elemental_data['r'])
    V0 = np.array(elemental_data['V0'])
    a = np.array(elemental_data['a'])
    a_prime = np.array(elemental_data['a_prime'])

    V0_unique = np.unique(V0)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ## First subplot: Axial (a)
    for v0 in V0_unique:
        mask = np.isclose(V0, v0)
        r_v0 = r[mask]
        a_v0 = a[mask]

        sort_idx = np.argsort(r_v0)
        axs[0].plot(r_v0[sort_idx], a_v0[sort_idx], '-', alpha=0.5,
                    label=f"V₀={v0:.1f} m/s")

    axs[0].set_ylabel('Axial Induction (a)')
    axs[0].set_title('Induction Factors vs Span (Grouped by V₀)')
    axs[0].grid(True)
    axs[0].legend(fontsize='small', ncol=2)

    ## Second subplot: Tangential (a')
    for v0 in V0_unique:
        mask = np.isclose(V0, v0)
        r_v0 = r[mask]
        a_prime_v0 = a_prime[mask]

        sort_idx = np.argsort(r_v0)
        axs[1].plot(r_v0[sort_idx], a_prime_v0[sort_idx], '--', alpha=0.5,
                    label=f"V₀={v0:.1f} m/s")

    axs[1].set_xlabel('Span Position r [m]')
    axs[1].set_ylabel('Tangential Induction (a\')')
    axs[1].grid(True)
    axs[1].legend(fontsize='small', ncol=2)

    plt.tight_layout()
    plt.show()

def plot_induction_vs_v0(solver):
    """
    Plots the axial and tangential induction factors against wind speed.
    Parameters:
        solver (object): An object containing simulation results and
        elemental data.
    Returns:
        None: Displays a plot of induction factors versus wind speed.
    """
    V0_list, a_list, a_prime_list = [], [], []
    for res in solver.results:
        V0 = res['V_0']
        mask = np.isclose(solver.elemental_data['V0'], V0)
        a_mean = np.mean(np.array(solver.elemental_data['a'])[mask])
        a_prime_mean = np.mean(np.array(
            solver.elemental_data['a_prime'])[mask])
        V0_list.append(V0)
        a_list.append(a_mean)
        a_prime_list.append(a_prime_mean)

    plt.figure(figsize=(8,6))
    plt.plot(V0_list, a_list, label="Axial (a)", marker='o')
    plt.plot(V0_list, a_prime_list, label="Tangential (a')", marker='x')
    plt.xlabel('Wind Speed V0 (m/s)')
    plt.ylabel('Induction Factors')
    plt.title('Induction Factors vs Wind Speed')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_induction_vs_pitch(solver):
    """
    Plots the induction factors (axial and tangential) against the pitch angle.
    Parameters:
        solver (object): An object containing simulation results and elemental
        data.
    Returns:
        None: Displays a plot of induction factors vs pitch angle.
    """
    pitch_list, a_list, a_prime_list = [], [], []
    for res in solver.results:
        pitch = res['pitch']
        mask = np.isclose(solver.elemental_data['V0'], res['V_0'])
        a_mean = np.mean(np.array(solver.elemental_data['a'])[mask])
        a_prime_mean = np.mean(np.array(
            solver.elemental_data['a_prime'])[mask])
        pitch_list.append(pitch)
        a_list.append(a_mean)
        a_prime_list.append(a_prime_mean)

    plt.figure(figsize=(8,6))
    plt.plot(pitch_list, a_list, label="Axial (a)", marker='o')
    plt.plot(pitch_list, a_prime_list, label="Tangential (a')", marker='x')
    plt.xlabel('Pitch Angle (θp) [deg]')
    plt.ylabel('Induction Factors')
    plt.title('Induction Factors vs Pitch Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_induction_vs_omega(solver):
    """
    Plots the induction factors (axial and tangential) against the
    rotational speed.
    Parameters:
        solver (object): An object containing simulation results and
        elemental data.
    Returns:
        None: Displays a plot of induction factors vs rotational speed.
    """
    omega_list, a_list, a_prime_list = [], [], []
    for res in solver.results:
        omega = res['omega']
        mask = np.isclose(solver.elemental_data['V0'], res['V_0'])
        a_mean = np.mean(np.array(solver.elemental_data['a'])[mask])
        a_prime_mean = np.mean(np.array(
            solver.elemental_data['a_prime'])[mask])
        omega_list.append(omega)
        a_list.append(a_mean)
        a_prime_list.append(a_prime_mean)

    plt.figure(figsize=(8,6))
    plt.plot(omega_list, a_list, label="Axial (a)", marker='o')
    plt.plot(omega_list, a_prime_list, label="Tangential (a')", marker='x')
    plt.xlabel('Rotational Speed (ω) [rad/s]')
    plt.ylabel('Induction Factors')
    plt.title('Induction Factors vs Rotational Speed')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cl_cd_vs_span(elemental_data):
    """
    Plots lift (Cl) and drag (Cd) coefficients versus span position (r),
    grouped by wind speed (V0).
    Parameters:
        elemental_data (dict): A dictionary containing the following keys:
            - 'r' (array-like): Span positions [m].
            - 'V0' (array-like): Wind speeds [m/s].
            - 'Cl' (array-like): Lift coefficients.
            - 'Cd' (array-like): Drag coefficients.
    Returns:
        None: Displays the plot.
    """
    r = np.array(elemental_data['r'])
    V0 = np.array(elemental_data['V0'])
    Cl = np.array(elemental_data['Cl'])
    Cd = np.array(elemental_data['Cd'])

    V0_unique = np.unique(V0)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ## First subplot: Cl
    for v0 in V0_unique:
        mask = np.isclose(V0, v0)
        r_v0 = r[mask]
        Cl_v0 = Cl[mask]

        sort_idx = np.argsort(r_v0)
        axs[0].plot(r_v0[sort_idx], Cl_v0[sort_idx], '-',
                    alpha=0.5, label=f"V₀={v0:.1f} m/s")

    axs[0].set_ylabel('Lift Coefficient Cl')
    axs[0].set_title('Lift Coefficient (Cl) vs Span (Grouped by V₀)')
    axs[0].grid(True)
    axs[0].legend(fontsize='small', ncol=2)

    ## Second subplot: Cd
    for v0 in V0_unique:
        mask = np.isclose(V0, v0)
        r_v0 = r[mask]
        Cd_v0 = Cd[mask]

        sort_idx = np.argsort(r_v0)
        axs[1].plot(r_v0[sort_idx], Cd_v0[sort_idx], '--', alpha=0.5,
                    label=f"V₀={v0:.1f} m/s")

    axs[1].set_xlabel('Span Position r [m]')
    axs[1].set_ylabel('Drag Coefficient Cd')
    axs[1].grid(True)
    axs[1].legend(fontsize='small', ncol=2)

    plt.tight_layout()
    plt.show()

def plot_cl_cd_vs_alpha(elemental_data):
    """
    Plots the Lift (Cl) and Drag (Cd) coefficients against the Angle of
    Attack (α) using two subplots.
    Parameters:
        elemental_data (dict): A dictionary containing 'alpha', 'Cl', and
        'Cd' as keys with corresponding numerical data.
    Returns:
        None: Displays the plot.
    """
    alpha = np.array(elemental_data['alpha'])
    Cl = np.array(elemental_data['Cl'])
    Cd = np.array(elemental_data['Cd'])

    # Optional: sort data by alpha for smooth curves
    sort_idx = np.argsort(alpha)
    alpha_sorted = alpha[sort_idx]
    Cl_sorted = Cl[sort_idx]
    Cd_sorted = Cd[sort_idx]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # First subplot: Cl vs alpha
    axs[0].plot(alpha_sorted, Cl_sorted, 'b-', alpha=0.7)
    axs[0].set_ylabel('Lift Coefficient Cl')
    axs[0].set_title('Lift Coefficient (Cl) vs Angle of Attack (α)')
    axs[0].grid(True)

    # Second subplot: Cd vs alpha
    axs[1].plot(alpha_sorted, Cd_sorted, 'r--', alpha=0.7)
    axs[1].set_xlabel('Angle of Attack α [deg]')
    axs[1].set_ylabel('Drag Coefficient Cd')
    axs[1].set_title('Drag Coefficient (Cd) vs Angle of Attack (α)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_moment_vs_v0(results):
    """
    Plots the relationship between wind speed (V0) and moment (M).
    Parameters:
        results (list of dict): A list of dictionaries where each
        dictionary contains 'V_0' (wind speed in m/s) and 'M' (moment in Nm).
    Returns:
        None: Displays a plot of Moment vs Wind Speed.
    """
    V0 = [res['V_0'] for res in results]
    M = [res['M'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(V0, M, marker='o')
    plt.xlabel('Wind Speed V0 (m/s)')
    plt.ylabel('Moment M (Nm)')
    plt.title('Moment vs Wind Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_thrust_vs_pitch(results):
    """
    Plots the thrust (T) versus pitch angle (θp) based on the given results.
    Parameters:
        results (list of dict): A list of dictionaries where each dictionary
        contains
            'pitch' (float) and 'T' (float) keys representing
            pitch angle and thrust values respectively.
    Returns:
        None: Displays a plot of thrust vs pitch angle.
    """
    pitch = [res['pitch'] for res in results]
    T = [res['T'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(pitch, T, marker='o')
    plt.xlabel('Pitch Angle θp (deg)')
    plt.ylabel('Thrust T (N)')
    plt.title('Thrust vs Pitch Angle')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_moment_vs_pitch(results):
    """
    Plots the relationship between pitch angle and moment.
    Parameters:
    results (list of dict): A list of dictionaries where each dictionary
    contains
        'pitch' (float) for pitch angle in degrees and
        'M' (float) for moment in Nm.
    Returns:
    None: Displays the plot.
    """
    pitch = [res['pitch'] for res in results]
    M = [res['M'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(pitch, M, marker='o')
    plt.xlabel('Pitch Angle θp (deg)')
    plt.ylabel('Moment M (Nm)')
    plt.title('Moment vs Pitch Angle')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_power_vs_pitch(results):
    """
    Plots the relationship between pitch angle and power.
    Parameters:
        results (list of dict): A list of dictionaries where each
        dictionary contains
            'pitch' (float) and 'P' (float) keys representing
            pitch angle and power, respectively.
    Returns:
        None: Displays a plot of power vs. pitch angle.
    """
    pitch = [res['pitch'] for res in results]
    P = [res['P'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(pitch, P, marker='o')
    plt.xlabel('Pitch Angle θp (deg)')
    plt.ylabel('Power P (W)')
    plt.title('Power vs Pitch Angle')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_thrust_vs_omega(results):
    """
    Plots the thrust (T) versus rotational speed (omega) for a wind turbine.
    Parameters:
        results (list of dict): A list of dictionaries where each dictionary
        contains
            'omega' (rotational speed in rad/s) and 'T' (thrust in N).
    Returns:
        None: Displays the plot.
    """
    omega = [res['omega'] for res in results]
    T = [res['T'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(omega, T, marker='o')
    plt.xlabel('Rotational Speed ω (rad/s)')
    plt.ylabel('Thrust T (N)')
    plt.title('Thrust vs Rotational Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_moment_vs_omega(results):
    """
    Plots the relationship between rotational speed (omega) and moment (M).
    Parameters:
        results (list of dict): A list of dictionaries where
        each dictionary contains
            'omega' (rotational speed in rad/s) and 'M' (moment in Nm).
    Returns:
        None: Displays a plot of moment vs rotational speed.
    """
    omega = [res['omega'] for res in results]
    M = [res['M'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(omega, M, marker='o')
    plt.xlabel('Rotational Speed ω (rad/s)')
    plt.ylabel('Moment M (Nm)')
    plt.title('Moment vs Rotational Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_power_vs_omega(results):
    """
    Plots the relationship between rotational speed (omega) and power (P).
    Parameters:
        results (list of dict): A list of dictionaries where each dictionary
                                contains 'omega' (rotational speed in rad/s)
                                and 'P' (power in watts).
    Returns:
        None: Displays the plot.
    """
    omega = [res['omega'] for res in results]
    P = [res['P'] for res in results]

    plt.figure(figsize=(8,6))
    plt.plot(omega, P, marker='o')
    plt.xlabel('Rotational Speed ω (rad/s)')
    plt.ylabel('Power P (W)')
    plt.title('Power vs Rotational Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tip_loss_vs_span(elemental_data):
    """
    Plots the Prandtl Tip Loss Factor (F) against the spanwise position (r),
    grouped by the freestream velocity (V₀).
    Parameters:
        elemental_data (dict): A dictionary containing:
            - 'r' (array-like): Spanwise positions [m].
            - 'V0' (array-like): Freestream velocities [m/s].
            - 'F' (array-like): Tip loss factors [-].
    Returns:
        None: Displays the plot.
    """
    r = np.array(elemental_data['r'])
    V0 = np.array(elemental_data['V0'])
    F = np.array(elemental_data['F'])

    V0_unique = np.unique(V0)

    plt.figure(figsize=(10, 6))

    for v0 in V0_unique:
        mask = np.isclose(V0, v0)
        r_v0 = r[mask]
        F_v0 = F[mask]

        # Sort by r to make smooth curves
        sort_idx = np.argsort(r_v0)
        r_v0_sorted = r_v0[sort_idx]
        F_v0_sorted = F_v0[sort_idx]

        plt.plot(r_v0_sorted, F_v0_sorted, '-', alpha=0.7,
                 label=f"V₀ = {v0:.1f} m/s")

    plt.xlabel('Spanwise Position r [m]')
    plt.ylabel('Tip Loss Factor F [-]')
    plt.title('Prandtl Tip Loss Factor vs Span (Grouped by V₀)')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

def plot_cp_ct_surfaces(results):
    """
    Plot CP and CT surfaces as a function of blade pitch (theta_p) and
    tip speed ratio (lambda).
    Parameters:
        results (list of dict): A list of dictionaries containing
        simulation data.
            Each dictionary must include the keys:
            'pitch' (float): Blade pitch angle in degrees,
            'omega' (float): Rotor angular velocity in rad/s,
            'V_0' (float): Wind speed in m/s,
            'C_P' (float): Power coefficient,
            'C_T' (float): Thrust coefficient.
    Returns:
        None: Displays the 3D surface plots for Cp and Ct.
    """

    # Step 1: Extract data
    pitch = np.array([res['pitch'] for res in results])   # degrees
    omega = np.array([res['omega'] for res in results])   # rad/s
    V0 = np.array([res['V_0'] for res in results])        # m/s
    cp = np.array([res['C_P'] for res in results])
    ct = np.array([res['C_T'] for res in results])

    # Step 2: Compute tip speed ratio lambda (λ = omega * R / V0)
    # (You need radius R; assuming it's imported or passed)
    # Assuming you have R defined in config
    from WindTurbineModeling.config import R
    lam = (omega * R) / V0

    # Step 3: Make surface plots
    fig = plt.figure(figsize=(16, 6))

    ## First subplot: Cp
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_trisurf(pitch, lam, cp, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Blade Pitch θp (deg)')
    ax1.set_ylabel('Tip Speed Ratio λ [-]')
    ax1.set_zlabel('Power Coefficient Cp')
    ax1.set_title('Power Coefficient (Cp) Surface')
    ax1.view_init(elev=30, azim=135)  # Nice viewing angle

    ## Second subplot: Ct
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_trisurf(pitch, lam, ct, cmap='plasma', edgecolor='none')
    ax2.set_xlabel('Blade Pitch θp (deg)')
    ax2.set_ylabel('Tip Speed Ratio λ [-]')
    ax2.set_zlabel('Thrust Coefficient Ct')
    ax2.set_title('Thrust Coefficient (Ct) Surface')
    ax2.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()
