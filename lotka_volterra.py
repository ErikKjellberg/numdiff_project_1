import matplotlib.pyplot as plt
import numpy as np

from RK34 import adaptiveRK34
from utils import plot_with_error

a = 3
b = 9
c = 15
d = 15
f = lambda t, y: np.array([a * y[0] - b * y[0] * y[1], c * y[0] * y[1] - d * y[1]])

y0 = np.array([1.5, 1.5])
t0 = 0
tf = 1
timegrid, approx, lerror = adaptiveRK34(f, y0, t0, tf, tol=1e-6)

plot_with_error(timegrid, approx, [np.linalg.norm(l) for l in lerror.transpose()])
plt.show()
