#TODO: Add docstrings to all functions and classes, commenting

import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bem_solvers import BEMSolver, BEMSolverOpt
from WindTurbineModeling.config import RATED_POWER, NUMBER_BLADES, R
from plot import (
    plot_induction_vs_span,
    plot_induction_vs_v0,
    plot_induction_vs_pitch,
    plot_induction_vs_omega,
    plot_cl_cd_vs_span,
    plot_cl_cd_vs_alpha,
    plot_moment_vs_v0,
    plot_thrust_vs_pitch,
    plot_moment_vs_pitch,
    plot_power_vs_pitch,
    plot_thrust_vs_omega,
    plot_moment_vs_omega,
    plot_power_vs_omega,
    plot_power_curve,
    plot_thrust_curve,
    plot_airfoil_shapes,   
    plot_wind_turbine,
    plot_tip_loss_vs_span,   
    plot_cp_ct_surfaces    
)

class ResultPlotter:
    def __init__(self):
        self.solver = None
        self.solver_type = None

    def run_solver(self):
        """Ask user to run Standard or Optimal solver"""
        print("\n🛠️ Choose Solver to Run:")
        print("1. Standard BEM Solver")
        print("2. Optimal Control BEM Solver")
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
        self.solver.run()
        end_time = time.time()
        print(f"✅ {self.solver_type} Solver Completed in {end_time - start_time:.2f} seconds.")

        # 👇 NEW: what to do after solving
        if self.solver_type == "Standard":
            self.plot_menu()
        else:  # Optimal
            self.optimal_control_menu()

    def optimal_control_menu(self):
        """Menu after Optimal Control Solver"""
        print("\n✅ Solver complete!")

        choice = input("📊 Do you want to visualize results? (y/n): ").lower().strip()

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
        """Interactive plotting menu after solver is complete"""
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
            print("19. Power and Thrust Coefficient Surfaces (Cp, Ct) vs Pitch and λ") 
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
                    print("⚠️ Invalid input. Enter a number, 's' to save, 'a' for all plots, or 'q' to quit.")


    def _generate_plot(self, choice):
        """Generate a plot based on user choice"""
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
            plot_power_curve(wind_speeds, [p/1e6 for p in powers], rated_power=RATED_POWER)
        elif choice == 18:
            plot_tip_loss_vs_span(self.solver.elemental_data)
        elif choice == 19:
            plot_cp_ct_surfaces(self.solver.results)
        else:
            print("⚠️ Invalid choice.")

    def _generate_all_plots(self):
        """Generate ALL available plots one after another"""
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
        plot_power_curve(wind_speeds, [p/1e6 for p in powers], rated_power=RATED_POWER)

        print("✅ Finished displaying all plots!")


    def _save_figure(self):
        """Save the current matplotlib figure"""
        filename = input("Enter filename (e.g., plot.png): ").strip()
        if not filename:
            filename = "bem_plot.png"
        save_path = Path("results") / filename
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figure saved to {save_path}")

    def plot_optimal_strategy(self):
        """Plot the 4 optimal control strategy plots as subplots and allow save/quit options"""

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
    plotter = ResultPlotter()
    plotter.run_solver()
    plotter.plot_menu()

if __name__ == "__main__":
    main()
