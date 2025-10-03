import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pysindy as ps
except ImportError:
    ps = None

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
print("pySINDy:", getattr(ps, "__version__", "NOT INSTALLED"))

# Tiny sanity check: simple plot
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
plt.plot(x, y)
plt.title("Matplotlib check: sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.tight_layout()
# Showing the figure will pop a window; comment the next line if running headless.
plt.show()
