import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


def plot_with_solution(timegrid, approx, solution):
    n, N = approx.shape
    plt.figure(1)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.scatter(timegrid, approx[i, :], label="Runge-Kutta", c="r", marker="x")
        plt.plot(timegrid, solution[i, :], label="Real solution")
        plt.legend()
        plt.yscale("log")
    plt.title("Approximation and solution curves")

    error = [
        np.linalg.norm(a - s) for a, s in zip(approx.transpose(), solution.transpose())
    ]
    plt.figure(2)
    plt.plot(timegrid, error)
    plt.yscale("log")
    plt.title("Error curves")


def plot_with_error(timegrid, approx, error):
    print(timegrid, approx, error)
    n, N = approx.shape
    plt.figure(1)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.scatter(timegrid, approx[i, :], label="Runge-Kutta", c="r", marker="x")
        plt.legend()
        plt.yscale("log")
    plt.title("Approximation and solution curves")

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
