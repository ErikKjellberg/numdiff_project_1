import matplotlib.pyplot as plt
import numpy as np

from utils import plot_with_solution, test_equation


def RK4step(f, told, uold, h):
    Y_1 = f(told, uold)
    Y_2 = f(told + h / 2, uold + h * Y_1 / 2)
    Y_3 = f(told + h / 2, uold + h * Y_2 / 2)
    Y_4 = f(told + h, uold + h * Y_3)
    return uold + h / 6 * (Y_1 + 2 * Y_2 + 2 * Y_3 + Y_4)


def RK4solver(f, y0, t0, tf, N):
    y = y0
    h = (tf - t0) / N
    approx = np.zeros((len(y0), N + 1))
    approx[:, 0] = y0
    t = t0
    timegrid = np.linspace(t0, tf, N + 1)
    for i in range(N):
        y = RK4step(f, t, y, h)
        approx[:, i + 1] = y
    return timegrid, approx


A = np.array([[-10, -1], [0, -1]])
f = lambda t, y: np.dot(A, y)
y0 = np.array([1, 1])
t0 = 0
tf = 1
N = 100
timegrid, approx = RK4solver(f, y0, t0, tf, N)
timegrid, sol = test_equation(A, y0, t0, tf, N)
plot_with_solution(
    timegrid,
    approx,
    [np.linalg.norm(a - s) for a, s in zip(approx.transpose(), sol.transpose())],
)

plt.show()
