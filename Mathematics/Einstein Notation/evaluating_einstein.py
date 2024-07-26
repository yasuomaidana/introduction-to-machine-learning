import time

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from numpy import tensordot, array
from einstein_mul_rs_lib import einstein_mul_rs


def rust_einstein_notation_mul(a, b):
    return array(einstein_mul_rs(a, b))


def einstein_notation_mul(a, b):
    return tensordot(a, b, ([-1], [-2]))


def plot_runtime_comparison(func_python, func_rust, title, xlabel, ylabel, x_values):
    y_values_python = []
    y_values_rust = []

    for size in tqdm(x_values):
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)

        start_time = time.time()
        func_python(a, b)
        end_time = time.time()
        y_values_python.append((end_time - start_time) * 1000)

        start_time = time.time()
        func_rust(a, b)
        end_time = time.time()
        y_values_rust.append((end_time - start_time) * 1000)

    plt.plot(x_values, y_values_python, label="Python")
    plt.plot(x_values, y_values_rust, label="Rust")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sizes = range(2, 500)
    plot_runtime_comparison(
        einstein_notation_mul,
        rust_einstein_notation_mul,
        "Runtime Comparison Python vs. Rust",
        "Matrix Size",
        "Time(ms)",
        sizes
    )
