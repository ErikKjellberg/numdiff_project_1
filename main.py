import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

print("Let's do approximations!")


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


def plot(timegrid, approx, solution):
    n, N = approx.shape
    plt.figure(1)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.plot(timegrid, approx[i, :], label="Runge-Kutta")
        plt.plot(timegrid, solution[i, :], label="Real solution")
        plt.legend()
        plt.yscale("log")
    plt.title("Approximation and solution curves")

    error = [
        np.linalg.norm(a - s) for a, s in zip(approx.transpose(), solution.transpose())
    ]
    print(error)
    plt.figure(2)
    plt.plot(timegrid, error)
    plt.yscale("log")
    plt.title("Error curves")


def test_equation(A, y0, t0, tf, N):
    solution = np.zeros((len(y0), N + 1))
    timegrid = np.linspace(t0, tf, N + 1)
    for i in range(N + 1):
        solution[:, i] = np.dot(linalg.expm(A * timegrid[i]), y0)
    return timegrid, solution


A = np.array([[-10, -1], [0, -1]])
f = lambda t, y: np.dot(A, y)
y0 = np.array([1, 1])
t0 = 0
tf = 1
N = 1000
timegrid, approx = RK4solver(f, y0, t0, tf, N)
timegrid, sol = test_equation(A, y0, t0, tf, N)
plot(timegrid, approx, sol)
plt.show()
