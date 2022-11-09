from math import pow

import matplotlib.pyplot as plt
import numpy as np

from utils import plot_with_error


def RK34step(f, told, yold, h):
    Y_1 = f(told, yold)
    Y_2 = f(told + h / 2, yold + h * Y_1 / 2)
    Y_3 = f(told + h / 2, yold + h * Y_2 / 2)
    Y_4 = f(told + h, yold + h * Y_3)
    Z_3 = f(told + h, yold - h * Y_1 + 2 * h * Y_2)
    ynew = yold + h / 6 * (Y_1 + 2 * Y_2 + 2 * Y_3 + Y_4)
    lnew = h / 6 * (2 * Y_2 + Z_3 - 2 * Y_3 - Y_4)
    return ynew, lnew


def newage(tol, r, rold, hold, k):
    return pow(tol / r, 2 / (3 * k)) * pow(tol / rold, -1 / (3 * k)) * hold


def RK34(f, y0, t0, tf, N):
    y = y0
    h = (tf - t0) / N
    approx = np.zeros((len(y0), N + 1))
    approx[:, 0] = y0
    lerror = np.zeros((len(y0), N))
    t = t0
    timegrid = np.linspace(t0, tf, N + 1)
    for i in range(N):
        y, l = RK34step(f, t, y, h)
        approx[:, i + 1] = y
        lerror[:, i] = l
    return timegrid, approx, lerror


def adaptiveRK34(f, y0, t0, tf, tol=1e-10):
    y = y0
    approx = [y0]
    lerror = [0]
    t = t0
    h = (tf - t0) * pow(tol, 1 / 4) / (100 * (1 + np.linalg.norm(f(t0, y0))))
    timegrid = [t0]
    t = t0
    i = 0
    rold = tol
    while t < tf:
        y, l = RK34step(f, t, y, h)
        r = np.linalg.norm(l)
        approx.append(y)
        lerror.append(l)
        t += h
        timegrid.append(t)
        h = newage(tol, r, rold, h, 4)
        rold = r
    return np.array(timegrid), np.array(approx).transpose(), np.array(lerror)


# A = np.array([[-10, -1], [0, -1]])
# f = lambda t, y: np.dot(A, y)
# y0 = np.array([1, 1])
# t0 = 0
# tf = 1
# N = 1000
# timegrid, approx, lerror = adaptiveRK34(f, y0, t0, tf, tol=1e-3)
# plot_with_error(timegrid, approx, [np.linalg.norm(l) for l in lerror.transpose()])
# plt.show()
