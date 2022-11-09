import matplotlib.pyplot as plt
import numpy as np

from RK34 import adaptiveRK34
from utils import plot_with_error

a = 3
b = 9
c = 15
d = 15
f = lambda t, y: np.array([a * y[0] - b * y[0] * y[1], c * y[0] * y[1] - d * y[1]])

y0 = np.array([1, 1])
t0 = 0
tf = 300
timegrid, approx, lerror = adaptiveRK34(f, y0, t0, tf, tol=1e-8)

H = lambda y: c * y[0] + b * y[1] - d * np.log(y[0]) - a * np.log(y[1])
H0 = H(y0)

plt.figure(10)
plt.plot(timegrid, [np.abs(H(y) / H0 - 1) for y in approx.transpose()])
plt.yscale("log")
"""
plot_with_error(timegrid, approx, [np.linalg.norm(l) for l in lerror.transpose()])
plt.figure(3)
plt.plot(approx[0, :], approx[1, :])
"""
plt.show()
