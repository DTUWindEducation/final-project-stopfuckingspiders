import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from io import StringIO

def plot_airfoil_shapes(airfoil_data_list, labels=None):
    plt.figure(figsize=(12, 6))
    for i, df in enumerate(airfoil_data_list):
        x, y = df["x/c"], df["y/c"]
        label = labels[i] if labels else f"AF{i:02d}"
        plt.plot(x, y, label=label, linewidth=1, alpha=0.7)
    plt.title("Airfoil Shapes")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(ncol=3, fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()