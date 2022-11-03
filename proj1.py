import matplotlib.pyplot as plt
import numpy as np


def RK4step(f, told, uold, h):
    Y1 = f(told, uold)
    Y2 = f(told + h / 2, uold + h * Y1 / 2)
    Y3 = f(told + h / 2, uold + h * Y2 / 2)
    Y4 = f(told + h, uold + h * Y3)
    return uold + h / 6 * (Y1 + 2 * Y2 + 2 * Y3 + Y4)


def RK4int(f, y0, t0, tf, N):
    h = (tf - t0) / N
    tgrid = np.linspace(t0, tf, N)
    approx = np.zeros((len(y0), N))
    u = y0
    t = t0
    for i in range(N):
        u = RK4step(f, t, u, h)
        approx[:, i] = u
        t += h
    return tgrid, approx

def RK4err()
    # For a 

l = 3.0
f = lambda t, y: l * y
y0 = np.array([1.0])
t0 = 0
tf = 1

tgrid, approx = RK4int(f, y0, t0, tf, 10)
print(tgrid, approx)
plt.plot(tgrid, approx[0, :])
plt.plot(tgrid, np.exp(tgrid * l))
plt.show()
