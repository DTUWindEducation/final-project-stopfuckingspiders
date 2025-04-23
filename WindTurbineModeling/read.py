import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas as pd
from pathlib import Path
#import config as c

#For Functions
import re
from typing import Union
from pathlib import Path

def read_blade_definition(file_path: Union[str, Path]):

    with open(file_path, "r") as file:
        content = file.readlines()
    
    # Column names from the header
    column_names = [
        "BlSpn", "BlCrvAC", "BlSwpAC", "BlCrvAng", "BlTwist", "BlChord",
        "BlAFID", "BlCb", "BlCenBn", "BlCenBt"
    ]
    
    # Extract only the lines with numeric data
    data_lines = [
        line for line in content
        if re.match(r"^\s*[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+", line)
    ]
    
    # Convert the lines into a DataFrame
    data = [list(map(float, line.split()[:10])) for line in data_lines]
    df = pd.DataFrame(data, columns=column_names)
    
    return df

def get_files_by_extension(folder_path: Union[str, Path], extensions: list[str] = [".csv"]) -> list[Path]:
    """
    Scans a folder (including sub-folders) and returns a list of file paths
    that match any of the specified extensions.

    Parameters:
        folder_path (Union[str, Path]): Path to the folder to scan.
        extensions (List[str], optional): List of file extensions to look for.
                                          Each extension should include the leading
                                          dot (e.g., '.csv'). Defaults to ['.csv'].

    Returns:
        List[Path]: List of matching file paths as Path objects.
    """
    folder_path = Path(folder_path)
    result = []

    for ext in extensions:
        # Ensure the extension starts with a dot
        if not ext.startswith("."):
            ext = f".{ext}"
        result.extend(folder_path.rglob(f"*{ext}"))

    return result

class WindTurbinePerformanceData:
    """
    Class for loading and parsing wind turbine performance data files.
    
    Handles data in the format:
    wind speed [m/s]   pitch [deg]   rot. speed [rpm]   aero power [kw]   aero thrust [kn]
    """
    
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.units = {}
        
    def load_file(self, filepath):
        """
        Load and parse a performance data file
        
        Args:
            filepath (str): Path to the data file
        """
        self.data = None
        self.metadata = {}
        self.units = {}
        
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            # Parse header line to get units
            if len(lines) > 0 and '[' in lines[0]:
                header_parts = lines[0].split()
                self._parse_units(header_parts)
                
            # Find where data starts (skip any additional header lines)
            data_start = 0
            for i, line in enumerate(lines):
                if line[0].strip().isdigit() or line.startswith(' '):
                    data_start = i
                    break
                    
            # Parse data rows
            data_rows = []
            for line in lines[data_start:]:
                # Clean line by removing extra spaces
                cleaned = ' '.join(line.split())
                parts = cleaned.split()
                
                # Convert to floats if possible
                converted = []
                for part in parts:
                    try:
                        converted.append(float(part))
                    except ValueError:
                        converted.append(part)
                
                if len(converted) == 5:  # We expect 5 columns
                    data_rows.append(converted)
                    
            # Create DataFrame
            columns = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
            self.data = pd.DataFrame(data_rows, columns=columns)
            
            # Add metadata
            self.metadata['num_points'] = len(self.data)
            self.metadata['min_wind_speed'] = self.data['wind_speed'].min()
            self.metadata['max_wind_speed'] = self.data['wind_speed'].max()
            
        except Exception as e:
            raise ValueError(f"Error parsing file: {str(e)}")
            
    def _parse_units(self, header_parts):
        """Parse units from header line"""
        # Header format: "wind speed [m/s] pitch [deg] rot. speed [rpm] ..."
        current_var = None
        for part in header_parts:
            if '[' in part:
                # This is a unit
                if current_var:
                    unit = part.strip('[]')
                    self.units[current_var] = unit
                    current_var = None
            else:
                # This is part of a variable name
                if part != 'speed' and part != 'aero':  # Skip connecting words
                    current_var = '_'.join([current_var, part]) if current_var else part
                    
        # Clean up variable names
        self.units = {
            'wind_speed': self.units.get('wind_speed', 'm/s'),
            'pitch': self.units.get('pitch', 'deg'),
            'rot_speed': self.units.get('rot_speed', 'rpm'),
            'aero_power': self.units.get('aero_power', 'kW'),
            'aero_thrust': self.units.get('aero_thrust', 'kN')
        }
        
    def get_power_curve(self):
        """Get wind speed vs power curve"""
        if self.data is not None:
            return self.data[['wind_speed', 'aero_power']].copy()
        return None
        
    def get_thrust_curve(self):
        """Get wind speed vs thrust curve"""
        if self.data is not None:
            return self.data[['wind_speed', 'aero_thrust']].copy()
        return None
        
    def plot_performance(self):
        """Create performance plots"""
        if self.data is None:
            print("No data loaded")
            return
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Power curve
        ax1.plot(self.data['wind_speed'], self.data['aero_power'], 'b-')
        ax1.set_title('Power Curve')
        ax1.set_xlabel(f"Wind Speed ({self.units.get('wind_speed', 'm/s')})")
        ax1.set_ylabel(f"Power ({self.units.get('aero_power', 'kW')})")
        ax1.grid(True)
        
        # Thrust curve
        ax2.plot(self.data['wind_speed'], self.data['aero_thrust'], 'r-')
        ax2.set_title('Thrust Curve')
        ax2.set_xlabel(f"Wind Speed ({self.units.get('wind_speed', 'm/s')})")
        ax2.set_ylabel(f"Thrust ({self.units.get('aero_thrust', 'kN')})")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Initialize parser
performance_data = WindTurbinePerformanceData()

# Load data file
file_path = r"C:\Users\Lenss\Desktop\second\protype\inputs\IEA-15-240-RWT\IEA_15MW_RWT_Onshore.opt"
performance_data.load_file(file_path)

# Access the data
print("Units:", performance_data.units)
print("Metadata:", performance_data.metadata)
print("First 5 rows:\n", performance_data.data.head(20))


# Get power curve data
power_curve = performance_data.get_power_curve()
print("Power curve:\n", power_curve.head())

# Get thrust curve data
thrust_curve = performance_data.get_thrust_curve()
print("Thrust curve:\n", thrust_curve.head())

# Plot the performance curves
performance_data.plot_performance()

