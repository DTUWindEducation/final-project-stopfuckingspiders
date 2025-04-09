# Function to plot airfoil shape
import matplotlib.pyplot as plt
import numpy as np  
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def load_airfoil(filename):
    """
    Load airfoil coordinates from a file.
    The file should contain x and y coordinates in two columns.
    """
    data = np.loadtxt(filename, skiprows=8)  # Skip header row if present
    x = data[:, 0]
    y = data[:, 1]
    return x, y
print("Loading airfoil data...")
x, y = load_airfoil('Shapes/IEA-15-240-RWT_AF20_Coords.txt')
print("Airfoil data loaded.")
# Plot the airfoil shape
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Airfoil Shape')
#plt.fill(x, y, alpha=0.5, label='Airfoil Fill')
plt.title('Airfoil Shape')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()