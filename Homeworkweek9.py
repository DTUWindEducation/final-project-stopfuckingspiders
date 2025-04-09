import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load LEANWIND 8MW power curve data
file_path = "LEANWIND_Reference_8MW_164.csv"  # Change this if needed
df_curve = pd.read_csv(file_path)
power_curve_data = df_curve[['Wind Speed [m/s]', 'Power [kW]']].to_numpy()

# General wind turbine class with cubic power model
class GeneralWindTurbine:
    def __init__(self, rotor_diameter, hub_height, rated_power,
                 v_in, v_rated, v_out, name=None):
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power  # in kW
        self.v_in = v_in
        self.v_rated = v_rated
        self.v_out = v_out
        self.name = name if name is not None else "Generic Turbine"

    def get_power(self, v):
        if v < self.v_in or v > self.v_out:
            return 0.0
        elif self.v_in <= v < self.v_rated:
            return self.rated_power * (v / self.v_rated) ** 3
        else:
            return self.rated_power

# Subclass using actual power curve data (interpolation)
class WindTurbine(GeneralWindTurbine):
    def __init__(self, rotor_diameter, hub_height, rated_power,
                 v_in, v_rated, v_out, power_curve_data, name=None):
        super().__init__(rotor_diameter, hub_height, rated_power,
                         v_in, v_rated, v_out, name)
        self.power_curve_data = power_curve_data

    def get_power(self, v):
        wind_speeds = self.power_curve_data[:, 0]
        powers = self.power_curve_data[:, 1]
        return float(np.interp(v, wind_speeds, powers))

# Define LEANWIND turbine specs
rotor_diameter = 164
hub_height = 110
rated_power = 8000     # kW
v_in = 4               # m/s
v_rated = 12.5         # m/s
v_out = 25             # m/s

# Instantiate both turbine models
generic_turbine = GeneralWindTurbine(
    rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name="General Model"
)

curve_turbine = WindTurbine(
    rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out,
    power_curve_data=power_curve_data, name="Interpolated Model"
)

# Generate power outputs over wind speed range
wind_speeds = np.linspace(0, 30, 300)
power_generic = [generic_turbine.get_power(v) for v in wind_speeds]
power_curve = [curve_turbine.get_power(v) for v in wind_speeds]

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(wind_speeds, power_generic, label="General Model", linestyle='--', color='orange')
plt.plot(wind_speeds, power_curve, label="Interpolated Model (LEANWIND data)", linestyle='-', color='blue')
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (kW)")
plt.title("Power Curve Comparison: General vs Interpolated Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
