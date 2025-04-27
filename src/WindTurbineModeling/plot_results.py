import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bem_solvers import BEMSolver, BEMSolverOpt

class ResultPlotter:
    def __init__(self):
        self.solvers = {
            'standard': None,
            'optimal': None
        }
    
    def load_or_run_solver(self, solver_type):
        """Load results from file or run solver"""
        choice = input(f"Load {solver_type} results from file? (y/n): ").lower()
        
        if choice == 'y':
            prefix = 'optimal_' if solver_type == 'optimal' else ''
            try:
                df = pd.read_csv(f"results/{prefix}performance_summary.csv")
                solver = BEMSolverOpt() if solver_type == 'optimal' else BEMSolver()
                solver.results = df.to_dict('records')
                print(f"✅ Loaded {solver_type} results successfully!")
                return solver
            except FileNotFoundError:
                print("⚠️ No saved results found. Running solver...")
                return self._run_solver(solver_type)
        else:
            return self._run_solver(solver_type)
    
    def _run_solver(self, solver_type):
        """Execute the specified solver"""
        solver = BEMSolverOpt() if solver_type == 'optimal' else BEMSolver()
        print(f"\nRunning {solver_type} BEM solver...")
        solver.run()
        return solver

    def get_available_plots(self):
        """List all possible plot combinations"""
        return {
            '1': ('Power Curve', ['power']),
            '2': ('Thrust Curve', ['thrust']),
            '3': ('Power Coefficient (CP)', ['cp']),
            '4': ('Thrust Coefficient (CT)', ['ct']),
            '5': ('Control Strategies', ['pitch', 'rpm']),
            '6': ('Compare Power', ['power_std', 'power_opt']),
            '7': ('Compare Thrust', ['thrust_std', 'thrust_opt']),
            '8': ('All Standard Plots', ['power', 'thrust', 'cp', 'ct']),
            '9': ('All Optimal Plots', ['power', 'thrust', 'pitch', 'rpm'])
        }

    def plot_interactively(self):
        """Interactive plotting menu"""
        print("\n" + "="*40)
        print("📊 BEM Result Visualization Tool")
        print("="*40)
        
        # Load solvers
        self.solvers['standard'] = self.load_or_run_solver('standard')
        self.solvers['optimal'] = self.load_or_run_solver('optimal')
        
        # Main menu
        while True:
            print("\n🔍 Available Plot Types:")
            plots = self.get_available_plots()
            for num, (name, _) in plots.items():
                print(f"{num}. {name}")
            
            print("\n⚙️ Options:")
            print("s. Save current figure")
            print("q. Quit")
            
            choice = input("\nSelect plot(s) or option: ").lower()
            
            if choice == 'q':
                break
            elif choice == 's':
                self._save_figure()
            elif choice in plots:
                self._generate_plot(plots[choice][1], title=plots[choice][0])
            else:
                print("⚠️ Invalid choice. Try again.")

    def _generate_plot(self, plot_types, title=""):
        """Generate the requested plots"""
        plt.figure(figsize=(12, 8))
        plt.suptitle(title, fontsize=14, y=1.02)
        
        for i, plot_type in enumerate(plot_types, 1):
            plt.subplot(2, 2, i)
            
            if plot_type == 'power':
                self._plot_curve('power', 'Power (MW)', 'b-')
            elif plot_type == 'thrust':
                self._plot_curve('thrust', 'Thrust (kN)', 'r-')
            # ... (add other plot types similarly)
            
            elif plot_type == 'power_std':
                self._plot_comparison('power', 'Standard', 'b-')
            elif plot_type == 'power_opt':
                self._plot_comparison('power', 'Optimal', 'r--')
            # ... (add other comparisons)
        
        plt.tight_layout()
        plt.show()

    def _plot_curve(self, metric, ylabel, style):
        """Plot a single curve"""
        for name, solver in self.solvers.items():
            if solver and hasattr(solver, 'results'):
                data = solver.get_plot_data()
                plt.plot(data['wind_speeds'], data[metric], style, 
                        label=name.capitalize())
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel(ylabel)
        plt.grid(True)
        if len(self.solvers) > 1:
            plt.legend()

    def _plot_comparison(self, metric, label, style):
        """Plot comparison between solvers"""
        if not all(self.solvers.values()):
            print("⚠️ Need both solvers for comparison")
            return
        
        std_data = self.solvers['standard'].get_plot_data()
        opt_data = self.solvers['optimal'].get_plot_data()
        
        plt.plot(std_data['wind_speeds'], std_data[metric], 'b-', 
                label='Standard')
        plt.plot(opt_data['wind_speeds'], opt_data[metric], 'r--', 
                label='Optimal')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()

    def _save_figure(self):
        """Save current figure to file"""
        filename = input("Enter filename (e.g., plot.png): ").strip()
        if not filename:
            filename = "bem_results.png"
        plt.savefig(Path("results") / filename, dpi=300, bbox_inches='tight')
        print(f"💾 Saved to {filename}")

if __name__ == "__main__":
    plotter = ResultPlotter()
    plotter.plot_interactively()