import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from WindTurbineModeling.plot import *

@pytest.fixture
def sample_elemental_data():
    return {
        'r': np.linspace(1, 10, 10),
        'V0': np.tile([5, 10], 5),
        'a': np.random.rand(10),
        'a_prime': np.random.rand(10),
        'Cl': np.random.rand(10),
        'Cd': np.random.rand(10),
        'F': np.random.rand(10),
        'alpha': np.linspace(-5, 15, 10)
    }

@pytest.fixture
def sample_results():
    return [
        {'V_0': 5, 'pitch': 2, 'omega': 1.2, 'M': 1000, 'T': 2000, 'P': 1500, 'C_P': 0.45, 'C_T': 0.85},
        {'V_0': 10, 'pitch': 4, 'omega': 1.4, 'M': 2000, 'T': 3000, 'P': 2500, 'C_P': 0.50, 'C_T': 0.80},
    ]

@pytest.fixture
def dummy_solver(sample_elemental_data, sample_results):
    class DummySolver:
        def __init__(self):
            self.results = sample_results
            self.elemental_data = sample_elemental_data
    return DummySolver()

def test_plot_airfoil_shapes():
    df1 = pd.DataFrame({'x/c': np.linspace(0, 1, 50), 'y/c': np.sin(np.linspace(0, 1, 50))})
    df2 = pd.DataFrame({'x/c': np.linspace(0, 1, 50), 'y/c': np.cos(np.linspace(0, 1, 50))})
    plot_airfoil_shapes([df1, df2], labels=["sin", "cos"])

def test_plot_wind_turbine():
    plot_wind_turbine()

def test_plot_power_curve():
    plot_power_curve(np.linspace(0, 25, 10), np.linspace(0, 3e6, 10), rated_power=2e6)

def test_plot_thrust_curve():
    plot_thrust_curve(np.linspace(0, 25, 10), np.linspace(0, 50000, 10))

def test_plot_cp_curve():
    plot_cp_curve(np.linspace(0, 25, 10), np.linspace(0.1, 0.5, 10))

def test_plot_ct_curve():
    plot_ct_curve(np.linspace(0, 25, 10), np.linspace(0.2, 0.9, 10))

def test_plot_induction_vs_span(sample_elemental_data):
    plot_induction_vs_span(sample_elemental_data)

def test_plot_induction_vs_v0(dummy_solver):
    plot_induction_vs_v0(dummy_solver)

def test_plot_induction_vs_pitch(dummy_solver):
    plot_induction_vs_pitch(dummy_solver)

def test_plot_induction_vs_omega(dummy_solver):
    plot_induction_vs_omega(dummy_solver)

def test_plot_cl_cd_vs_span(sample_elemental_data):
    plot_cl_cd_vs_span(sample_elemental_data)

def test_plot_cl_cd_vs_alpha(sample_elemental_data):
    plot_cl_cd_vs_alpha(sample_elemental_data)

def test_plot_moment_vs_v0(sample_results):
    plot_moment_vs_v0(sample_results)

def test_plot_thrust_vs_pitch(sample_results):
    plot_thrust_vs_pitch(sample_results)

def test_plot_moment_vs_pitch(sample_results):
    plot_moment_vs_pitch(sample_results)

def test_plot_power_vs_pitch(sample_results):
    plot_power_vs_pitch(sample_results)

def test_plot_thrust_vs_omega(sample_results):
    plot_thrust_vs_omega(sample_results)

def test_plot_moment_vs_omega(sample_results):
    plot_moment_vs_omega(sample_results)

def test_plot_power_vs_omega(sample_results):
    plot_power_vs_omega(sample_results)

def test_plot_tip_loss_vs_span(sample_elemental_data):
    plot_tip_loss_vs_span(sample_elemental_data)

def test_plot_cp_curve():
    plot_cp_curve(np.linspace(0, 25, 10), np.linspace(0.1, 0.5, 10))
    plt.close("all")

