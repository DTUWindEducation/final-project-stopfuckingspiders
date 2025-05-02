import pytest
import matplotlib
matplotlib.use("Agg")  # Disable GUI for plotting
import matplotlib.pyplot as plt
from pathlib import Path


from WindTurbineModeling.interactive import ResultPlotter


@pytest.fixture
def dummy_solver():
    class DummySolver:
        def __init__(self):
            self.results = [
                {'V_0': 5, 'T': 2000, 'P': 1e6, 'M': 3000, 'pitch': 2, 'omega': 10, 'C_P': 0.45, 'C_T': 0.85},
                {'V_0': 10, 'T': 4000, 'P': 2e6, 'M': 6000, 'pitch': 4, 'omega': 15, 'C_P': 0.50, 'C_T': 0.82}
            ]
            self.elemental_data = {
                'r': [1, 2], 'V0': [5, 10], 'a': [0.2, 0.3], 'a_prime': [0.01, 0.02],
                'Cl': [0.5, 0.6], 'Cd': [0.01, 0.015], 'F': [0.98, 0.95], 'alpha': [-2, 5]
            }
            self.inputs = {'airfoil_shapes': []}

        def run(self):
            pass

    return DummySolver()

def test_run_solver_standard(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")  # Choose Standard solver
    plotter = ResultPlotter()
    monkeypatch.setattr(ResultPlotter, "plot_menu", lambda self: None)
    plotter.run_solver()
    assert plotter.solver_type == "Standard"
    assert plotter.solver is not None

def test_run_solver_optimal(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")  # Choose Optimal solver
    plotter = ResultPlotter()
    monkeypatch.setattr(ResultPlotter, "optimal_control_menu", lambda self: None)
    plotter.run_solver()
    assert plotter.solver_type == "Optimal"
    assert plotter.solver is not None

def test_generate_all_plots(dummy_solver):
    plotter = ResultPlotter()
    plotter.solver = dummy_solver
    plotter._generate_all_plots()
    plt.close("all")

def test_generate_plot_valid(dummy_solver):
    plotter = ResultPlotter()
    plotter.solver = dummy_solver
    for i in range(1, 20):  # test all valid plot choices
        plotter._generate_plot(i)
    plt.close("all")

def test_generate_plot_invalid(dummy_solver, capsys):
    plotter = ResultPlotter()
    plotter.solver = dummy_solver
    plotter._generate_plot(999)  # Invalid choice
    captured = capsys.readouterr()
    assert "Invalid choice" in captured.out

def test_save_figure(tmp_path, monkeypatch):
    plotter = ResultPlotter()
    plt.plot([0, 1], [0, 1])

    monkeypatch.setattr("builtins.input", lambda _: "test_plot.png")

    # Patch Path to redirect "results" to tmp_path
    monkeypatch.setattr("WindTurbineModeling.interactive.Path", lambda p="": tmp_path)
    
    plotter._save_figure()
    assert (tmp_path / "test_plot.png").exists()
    plt.close("all")

def test_plot_optimal_strategy(dummy_solver):
    plotter = ResultPlotter()
    plotter.solver = dummy_solver
    plotter.plot_optimal_strategy()
    plt.close("all")
