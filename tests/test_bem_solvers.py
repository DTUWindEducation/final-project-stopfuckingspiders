import os
import pandas as pd
import numpy as np
from WindTurbineModeling.bem_solvers import BaseBEMSolver
from collections import defaultdict


def test_solver_initializes_empty():
    solver = BaseBEMSolver()
    assert solver.results is None
    assert solver.inputs is None
    assert solver.elemental_data is None


def test_solve_blade_element_minimal_valid():
    solver = BaseBEMSolver()

    # Provide simple, valid input
    result = solver.solve_blade_element(
        V0=10,
        omega=1.0,
        r=5,
        pitch=0,
        beta=0,
        chord=1.0,
        df_polar=pd.DataFrame({
            "Alpha (deg)": [0, 5, 10],
            "Cl": [0.5, 1.0, 1.2],
            "Cd": [0.01, 0.02, 0.03]
        }),
        num_blades=3,
        radius=10,
        rho=1.225,
        max_iter=5,
        tolerance=1e-6,
        af_id=1
    )

    assert result is not None
    assert isinstance(result['p_n'], float)
    assert isinstance(result['p_t'], float)


def test_get_plot_data_simple():
    solver = BaseBEMSolver()
    solver.results = [
        {"V_0": 5, "P": 1e5, "T": 1000, "C_P": 0.4, "C_T": 0.8},
        {"V_0": 10, "P": 2e5, "T": 2000, "C_P": 0.5, "C_T": 0.9}
    ]
    data = solver.get_plot_data()
    assert "wind_speeds" in data
    assert len(data["power"]) == 2
    assert data["power"][0] == 0.1


def test_save_results(tmp_path):
    solver = BaseBEMSolver()

    results = [{"V_0": 5, "P": 100000, "T": 1000, "C_P": 0.45, "C_T": 0.8}]
    elementals = [{"r": 5, "alpha": 10}]
    airfoil_data = {
        1: pd.DataFrame({"Alpha": [0, 5], "Cl": [0.5, 1.0]})
    }

    # Temporarily change working dir
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        solver.save_results(results, elementals, airfoil_data)
        assert (tmp_path / "results/performance_summary.csv").exists()
        assert (tmp_path / "results/elemental_data.csv").exists()
        assert (tmp_path / "results/airfoils/airfoil_1.csv").exists()
    finally:
        os.chdir(old_cwd)


def test_solve_blade_element():
    solver = BaseBEMSolver()
    V0 = 10.0
    omega = 1.0
    r = 5.0
    pitch = 2.0
    beta = 0.0
    chord = 1.0
    df_polar = pd.DataFrame({'Alpha (deg)': [0], 'Cl': [1.0], 'Cd': [0.01]})
    num_blades = 3
    radius = 50.0
    rho = 1.225
    max_iter = 1
    tolerance = 0.001
    af_id = 1

    result = solver.solve_blade_element(V0, omega, r, pitch, beta, chord, df_polar,
                                        num_blades, radius, rho, max_iter, tolerance, af_id)
    assert result is not None
    assert 'p_n' in result
    assert 'p_t' in result
    assert 'elemental' in result


def test_get_plot_data():
    solver = BaseBEMSolver()
    solver.results = [{'V_0': 5.0, 'P': 1000.0, 'T': 500.0, 'C_P': 0.4, 'C_T': 0.8}]
    data = solver.get_plot_data()
    assert 'wind_speeds' in data
    assert 'power' in data
    assert 'thrust' in data
    assert 'cp' in data
    assert 'ct' in data
