import matplotlib.pyplot as plt
import numpy as np

from RK34 import adaptiveRK34
from utils import plot_with_error

# Good values for mu: 10,15,22,33,47,68,100,150,220,330,470,680,1000
mu = 1
f = lambda t, y: np.array([y[1], mu * (1 - -y[0] ** 2) * y[1] - y[0]])
y0 = np.array([2, 0])
t0 = 0
tf = 0.7 * mu
timegrid, approx, lerror = adaptiveRK34(f, y0, t0, tf, 1e-10)
plot_with_error(timegrid, approx, [np.linalg.norm(l) for l in lerror.transpose()])
plt.figure(3)
plt.plot(approx[0, :], approx[1, :])
plt.show()
