import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from WindTurbineModeling.bem_solvers import BEMSolver, BEMSolverOpt
from WindTurbineModeling.config import RATED_POWER, NUMBER_BLADES, R
from WindTurbineModeling.plot import *

class ResultPlotter:
    """
    ResultPlotter is a class designed to interactively run solvers for wind
    turbine modeling and visualize the results through various plots.
    It provides options for running a Standard BEM Solver or an Optimal Control
    BEM Solver, and includes menus for generating specific plots,
    saving figures, and restarting the solver.
    Methods:
    ---------
    __init__():
        Initializes the ResultPlotter instance with default solver
        and solver type.
    run_solver():
        Prompts the user to choose and run either the Standard or
        Optimal Control BEM Solver.
        Displays progress and execution time, and navigates to the appropriate
        menu based on
        the solver type.
    optimal_control_menu():
        Provides a menu for interacting with results after running the Optimal
        Control Solver.
        Allows visualization of results, saving figures, restarting the solver,
        or exiting.
    plot_menu():
        Provides an interactive menu for generating various plots after running
        the Standard
        BEM Solver. Includes options for restarting the solver, saving figures,
        generating
        all plots, or exiting.
    _generate_plot(choice):
        Generates a specific plot based on the user's choice from the plot menu.
    _generate_all_plots():
        Generates all available plots sequentially, including geometry,
        BEM-related plots,
        and performance curves.
    _save_figure():
        Saves the current matplotlib figure to a specified file in the
        "results" directory.
    plot_optimal_strategy():
        Generates and displays a 2x2 grid of subplots showing the optimal
        control strategy results, including pitch vs wind speed,
        RPM vs wind speed, power vs wind speed, and thrust vs wind speed.
    """
    def __init__(self):
        """
        Initializes the instance with default solver and
        solver type set to None.
        """
        self.solver = None
        self.solver_type = None

    def run_solver(self):
        """
        Runs the selected solver (Standard or Optimal) based on user input.
        Prompts the user to choose between the Standard BEM Solver or the
        Optimal Control BEM Solver, executes the chosen solver, and displays
        the execution time. Provides additional options based
        on the solver type.
        """
        print("\n Choose Solver to Run:")
        print("1. Standard BEM Solver (Functional requirements 1-5)")
        print("2. Optimal Control BEM Solver (Functional requirements 6-7)")
        choice = input("Enter 1 or 2: ").strip()

        if choice == '1':
            self.solver = BEMSolver()
            self.solver_type = "Standard"
        elif choice == '2':
            self.solver = BEMSolverOpt()
            self.solver_type = "Optimal"
        else:
            print("⚠️ Invalid choice. Exiting.")
            exit(1)

        print(f"\n🚀 Running {self.solver_type} BEM Solver...")
        start_time = time.time()
        self.solver.run()  # will print step-by-step progress internally
        end_time = time.time()
        print(f"✅ {self.solver_type} \
              Solver completed in {end_time - start_time:.2f} seconds.\n")

        if self.solver_type == "Standard":
            self.plot_menu()
        else:
            self.optimal_control_menu()


    def optimal_control_menu(self):
        """
        Provides a menu for interacting with results after running the
        Optimal Control Solver.

        Inputs:
        -------
        - User input for visualizing results ('y' or 'n').
        - User input for menu options ('s', 'r', or 'q').

        Outputs:
        --------
        - Displays the optimal control strategy plots if chosen.
        - Saves the current figure to a file if requested.
        - Restarts the solver or exits the program based on user choice.
        """
        print("\n✅ Solver complete!")

        choice = input("📊 Do you want to visualize results? (y/n): "
                       ).lower().strip()

        if choice == 'y':
            self.plot_optimal_strategy()

        while True:
            print("\n⚙️ Options:")
            print("s. Save current figure")
            print("r. Restart and choose new solver")
            print("q. Quit")

            option = input("Enter your choice: ").lower().strip()

            if option == 's':
                self._save_figure()
            elif option == 'r':
                self.run_solver()
                break
            elif option == 'q':
                print("👋 Exiting. Bye!")
                exit(0)
            else:
                print("⚠️ Invalid input. Try again.")


    def plot_menu(self):
        """
        Displays an interactive plotting menu for visualizing wind turbine
        simulation results. This method provides a menu-driven interface for
        users to select various plotting options after the solver has completed
        its computations. Users can generate specific plots, save figures,
        visualize all plots, or restart the solver.
        Input:
            - User input via the console to select menu options:
                * 'q': Quit the menu.
                * '0': Restart the solver.
                * 's': Save the current figure.
                * 'a': Generate and display all plots.
                * Integer (0-19): Generate a specific plot based on the
                selected option.
        Output:
            - Executes the corresponding action based on user input:
                * Generates and displays plots.
                * Saves the current figure to a file.
                * Restarts the solver.
                * Exits the menu.
            - Prints messages to guide the user and indicate the status
            of actions.
        """
        while True:
            print("\n📈 Solver complete! What do you want to do?")
            print("\nSelect an option:")
            print("0. 🔄 Change Solver (restart)")
            print("1. Plot Airfoil Shapes")
            print("2. Plot Wind Turbine Structure")
            print("3. Induction Factors vs Span (r)")
            print("4. Induction Factors vs Wind Speed (V0)")
            print("5. Induction Factors vs Pitch Angle (θp)")
            print("6. Induction Factors vs Rotational Speed (ω)")
            print("7. Cl and Cd vs Span (r)")
            print("8. Cl and Cd vs Angle of Attack (α)")
            print("9. Moment vs Wind Speed (V0)")
            print("10. Thrust vs Pitch Angle (θp)")
            print("11. Moment vs Pitch Angle (θp)")
            print("12. Power vs Pitch Angle (θp)")
            print("13. Thrust vs Rotational Speed (ω)")
            print("14. Moment vs Rotational Speed (ω)")
            print("15. Power vs Rotational Speed (ω)")
            print("16. Thrust vs Wind Speed (V0)")
            print("17. Power vs Wind Speed (V0)")
            print("18. Prandtl Tip Loss Factor vs Radius r")
            print("19. Power and Thrust Coefficient Surfaces (Cp, Ct) vs " \
            "Pitch and λ")
            print("\na. 📊 Visualize ALL plots")
            print("\ns. 💾 Save current figure")
            print("q. ❌ Quit")

            choice = input("\nEnter your choice: ").lower().strip()

            if choice == 'q':
                print("👋 Exiting. Bye!")
                break
            elif choice == '0':
                print("🔄 Restarting solver...")
                self.run_solver()
            elif choice == 's':
                self._save_figure()
            elif choice == 'a':
                self._generate_all_plots()
            else:
                try:
                    choice = int(choice)
                    self._generate_plot(choice)
                except ValueError:
                    print("⚠️ Invalid input. Enter a number, 's' to save,"
                    " 'a' for all plots, or 'q' to quit.")


    def _generate_plot(self, choice):
        """
        Generate and display a specific plot based on the user's choice.

        Parameters:
            choice (int): An integer representing the user's selection for the
                          type of plot to generate. Valid choices range
                          from 1 to 19.

        Outputs:
            Displays the selected plot. If the choice is invalid, a warning
            message
            is printed instead.
        """
        if choice == 1:
            plot_airfoil_shapes(self.solver.inputs['airfoil_shapes'])
        elif choice == 2:
            plot_wind_turbine()
        elif choice == 3:
            plot_induction_vs_span(self.solver.elemental_data)
        elif choice == 4:
            plot_induction_vs_v0(self.solver)
        elif choice == 5:
            plot_induction_vs_pitch(self.solver)
        elif choice == 6:
            plot_induction_vs_omega(self.solver)
        elif choice == 7:
            plot_cl_cd_vs_span(self.solver.elemental_data)
        elif choice == 8:
            plot_cl_cd_vs_alpha(self.solver.elemental_data)
        elif choice == 9:
            plot_moment_vs_v0(self.solver.results)
        elif choice == 10:
            plot_thrust_vs_pitch(self.solver.results)
        elif choice == 11:
            plot_moment_vs_pitch(self.solver.results)
        elif choice == 12:
            plot_power_vs_pitch(self.solver.results)
        elif choice == 13:
            plot_thrust_vs_omega(self.solver.results)
        elif choice == 14:
            plot_moment_vs_omega(self.solver.results)
        elif choice == 15:
            plot_power_vs_omega(self.solver.results)
        elif choice == 16:
            wind_speeds = [res['V_0'] for res in self.solver.results]
            thrusts = [res['T'] for res in self.solver.results]
            plot_thrust_curve(wind_speeds, thrusts)
        elif choice == 17:
            wind_speeds = [res['V_0'] for res in self.solver.results]
            powers = [res['P'] for res in self.solver.results]
            plot_power_curve(wind_speeds, [p/1e6 for p in powers],
                             rated_power=RATED_POWER)
        elif choice == 18:
            plot_tip_loss_vs_span(self.solver.elemental_data)
        elif choice == 19:
            plot_cp_ct_surfaces(self.solver.results)
        else:
            print("⚠️ Invalid choice.")

    def _generate_all_plots(self):
        """
        Generate and display all available plots for wind turbine analysis.
        This method sequentially generates various plots related to the
        geometry,        Blade Element Momentum (BEM) analysis, and performance
        curves of the wind turbine.
        Inputs:
        - self: The instance of the class containing solver data and inputs.
        Outputs:
        - None: The method generates and displays plots but does not return
        any value.
        Plots Generated:
        - Geometry-related plots:
            * Airfoil shapes
            * Wind turbine geometry
        - BEM-related plots:
            * Induction factor vs span
            * Induction factor vs wind speed (V_0)
            * Induction factor vs pitch angle
            * Induction factor vs rotational speed (omega)
            * Lift-to-drag ratio (Cl/Cd) vs span
            * Lift-to-drag ratio (Cl/Cd) vs angle of attack (alpha)
            * Moment vs wind speed (V_0)
            * Thrust vs pitch angle
            * Moment vs pitch angle
            * Power vs pitch angle
            * Thrust vs rotational speed (omega)
            * Moment vs rotational speed (omega)
            * Power vs rotational speed (omega)
        - Performance curves:
            * Thrust curve (Thrust vs wind speed)
            * Power curve (Power vs wind speed, normalized to MW)
        Note:
        - The method assumes that the solver contains precomputed results
        and inputs.
        - The rated power for the power curve is specified by the constant
        `RATED_POWER`.
        """

        print("\n📊 Generating all plots...")

        # Geometry
        plot_airfoil_shapes(self.solver.inputs['airfoil_shapes'])
        plot_wind_turbine()

        # BEM related
        plot_induction_vs_span(self.solver.elemental_data)
        plot_induction_vs_v0(self.solver)
        plot_induction_vs_pitch(self.solver)
        plot_induction_vs_omega(self.solver)
        plot_cl_cd_vs_span(self.solver.elemental_data)
        plot_cl_cd_vs_alpha(self.solver.elemental_data)
        plot_moment_vs_v0(self.solver.results)
        plot_thrust_vs_pitch(self.solver.results)
        plot_moment_vs_pitch(self.solver.results)
        plot_power_vs_pitch(self.solver.results)
        plot_thrust_vs_omega(self.solver.results)
        plot_moment_vs_omega(self.solver.results)
        plot_power_vs_omega(self.solver.results)

        # Curves
        wind_speeds = [res['V_0'] for res in self.solver.results]
        thrusts = [res['T'] for res in self.solver.results]
        powers = [res['P'] for res in self.solver.results]

        plot_thrust_curve(wind_speeds, thrusts)
        plot_power_curve(wind_speeds, [p/1e6 for p in powers],
                         rated_power=RATED_POWER)

        print("✅ Finished displaying all plots!")


    def _save_figure(self):
        """
        Save the current matplotlib figure to a file.

        This method prompts the user to enter a filename for saving the current
        matplotlib figure. If no filename is provided, a default name
        ("bem_plot.png") is used. The figure is saved in the "results"
        directory, which is created if it does not already exist.

        Input:
            - User input for the filename (e.g., "plot.png"). If left blank,
              defaults to "bem_plot.png".

        Output:
            - Saves the figure as a file in the "results" directory with the
              specified or default filename.
            - Prints a confirmation message with the save path.
        """
        filename = input("Enter filename (e.g., plot.png): ").strip()
        if not filename:
            filename = "bem_plot.png"
        save_path = Path("results") / filename
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figure saved to {save_path}")

    def plot_optimal_strategy(self):
        """
        Plot the optimal control strategy results for a wind turbine model.
        This method generates a 2x2 grid of subplots to visualize the
        relationships between wind speed and key turbine parameters:
        pitch angle, rotational speed (RPM), power output, and thrust force.
        Each subplot includes appropriate labels, titles,
        and grid lines for clarity.
        Inputs:
        - None: This method uses the `self.solver.results` attribute,
        which is expected to be a list of dictionaries containing the
        following keys:
          - 'V_0': Wind speed (m/s)
          - 'Pitch': Pitch angle (degrees)
          - 'RPM': Rotational speed (RPM)
          - 'P': Power output (Watts)
          - 'T': Thrust force (Newtons)
        Outputs:
        - None: Displays the plots using Matplotlib's `plt.show()` method.
        """
        V0 = [res['V_0'] for res in self.solver.results]
        pitch = [res['Pitch'] for res in self.solver.results]
        rpm = [res['RPM'] for res in self.solver.results]
        power = [res['P']/1e6 for res in self.solver.results]  # MW
        thrust = [res['T']/1000 for res in self.solver.results]  # kN

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Optimal Control Strategy Results', fontsize=16)

        # 1. Pitch vs Wind Speed
        axs[0,0].plot(V0, pitch, 'm-o')
        axs[0,0].set_xlabel('Wind Speed V₀ (m/s)')
        axs[0,0].set_ylabel('Pitch Angle θp (deg)')
        axs[0,0].set_title('Pitch Control Strategy')
        axs[0,0].grid(True)

        # 2. RPM vs Wind Speed
        axs[0,1].plot(V0, rpm, 'c-s')
        axs[0,1].set_xlabel('Wind Speed V₀ (m/s)')
        axs[0,1].set_ylabel('Rotational Speed ω (RPM)')
        axs[0,1].set_title('RPM Control Strategy')
        axs[0,1].grid(True)

        # 3. Power vs Wind Speed
        axs[1,0].plot(V0, power, 'b-^')
        axs[1,0].set_xlabel('Wind Speed V₀ (m/s)')
        axs[1,0].set_ylabel('Power P (MW)')
        axs[1,0].set_title('Power Curve')
        axs[1,0].grid(True)

        # 4. Thrust vs Wind Speed
        axs[1,1].plot(V0, thrust, 'g-x')
        axs[1,1].set_xlabel('Wind Speed V₀ (m/s)')
        axs[1,1].set_ylabel('Thrust T (kN)')
        axs[1,1].set_title('Thrust Curve')
        axs[1,1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def main():
    """
    Entry point for the interactive wind turbine modeling application.
    """
    plotter = ResultPlotter()
    plotter.run_solver()
    plotter.plot_menu()

if __name__ == "__main__":
    # Call the main function to start the interactive wind turbine
    # modeling application.
    main()
