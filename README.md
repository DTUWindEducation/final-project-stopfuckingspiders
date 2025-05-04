# Wind Turbine Modelling

Team: [Stop Fucking Spiders]

## Overview

**WindTurbineModeling** is a Python package for simulating the aerodynamic performance of horizontal-axis wind turbines using a steady-state Blade Element Momentum (BEM) model. The objective of this project is to compute key performance metrics—such as power output, thrust, and torque—as functions of wind speed, rotor speed, and blade pitch angle. The package includes tools for loading turbine geometry, solving BEM equations, and visualizing performance across a range of conditions.

## Quick-start guide

### 1. Clone the Repository

```bash
git clone https://github.com/DTUWindEducation/final-project-stopfuckingspiders.git
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv windturbine
source windturbine/bin/activate  # On Windows use: windturbine\Scripts\activate
```

### 3. Install the Package in Editable Mode

Make sure you're in the root directory (where `pyproject.toml` is):

```bash
pip install -e .
```

### 4. Run the Interactive Application

```bash
python examples/main.py
```

## Architecture

[ADD TEXT HERE!]



## Configuration

- **`config.yaml`** (located in the project root):  
  Stores all key project settings, including folder paths, turbine parameters, and numerical constants.

- **`config.py`** (located in `src/WindTurbineModeling/`):  
  Loads values from `config.yaml` and defines accessible variables for use throughout the codebase. It also sets up absolute paths relative to the project root for input and output data.

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`


# Wind Turbine Modeling – Examples

This folder `examples/` contains usage examples for running the Wind Turbine Modeling package, including an illustrative diagrams of the simulation workflow.

## Files

- **`main.py`**  
  Launches an interactive simulation interface. Allows users to choose between:
  - Standard BEM Solver
  - Optimal Control BEM Solver

  **To run:**
  ```bash
  python examples/main.py
  ```

  The solver performs a complete wind turbine BEM analysis, loading geometry and performance data, and outputs key metrics like thrust, torque, and power. It also provides a suite of plotting functions for visualizing results.

## Features Demonstrated

- Loading and preprocessing turbine data
- Airfoil and performance plotting
- Induction factor calculations
- Optimal pitch and speed estimation
- Power and thrust curve generation
- Interactive re-execution and figure saving



## Documentation

The folder `docs` contains the official manual from the U.S. Department of Energy’s National Renewable Energy Laboratory (**`aerodyn_v15_user_guide_and_theory_manual.pdf`**). The document contains detailed descriptions of abbreviation of variables used in the model. Further this guide includes:
  - Detailed descriptions of aerodynamic modeling options
  - Input/output file specifications
  - Blade Element Momentum (BEM) theory implementation
  - Unsteady airfoil aerodynamics (Beddoes-Leishman models)
  - Configuration options for standalone and FAST-coupled simulations
