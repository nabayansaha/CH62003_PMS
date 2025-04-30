import numpy as np
import matplotlib.pyplot as plt
import time as tm

# Problem parameters
L = 0.01         # Length of the rod
T0 = 300         # Initial and boundary temperature
a = 443.42       # Thermal conductivity constant
b = 0.04675      # Temperature coefficient
omega = 1        # Frequency of boundary temp variation
dt = 0.01        # Time step
rho = 8000       # Density (kg/m^3)
Cp = 500         # Specific heat (J/kg·K)
n_cycles = 5
time = np.arange(0, n_cycles / omega, dt)

# Function: temperature-dependent thermal conductivity
def k(T_val):
    return a - b * T_val

# Function: Thomas Algorithm for tridiagonal system
def thomas_algorithm(a_, b_, c_, d_):
    n = len(d_)
    a_copy = np.copy(a_)
    b_copy = np.copy(b_)
    c_copy = np.copy(c_)
    d_copy = np.copy(d_)
    
    for i in range(1, n):
        w = a_copy[i-1] / b_copy[i-1]
        b_copy[i] -= w * c_copy[i-1]
        d_copy[i] -= w * d_copy[i-1]
    
    x = np.zeros(n)
    x[-1] = d_copy[-1] / b_copy[-1]
    for i in reversed(range(n-1)):
        x[i] = (d_copy[i] - c_copy[i] * x[i+1]) / b_copy[i]
    
    return x
# ----------------------- Part (e): Time Comparison ----------------------
grid_sizes = [10, 50, 100, 200, 500, 1000,10000]
thomas_times = []
gauss_times = []

for N in grid_sizes:
    dx_test = L / N
    n_grid_test = N + 1
    num_internal = n_grid_test - 2

    T_test = np.full((n_grid_test, 2), T0)
    k_val = k(T0)
    lam = k_val * dt / (rho * Cp * dx_test**2)

    a_diag = -lam * np.ones(num_internal - 1)
    b_diag = (1 + 2 * lam) * np.ones(num_internal)
    c_diag = -lam * np.ones(num_internal - 1)
    d_rhs = T0 * np.ones(num_internal)
    d_rhs[0] += lam * T0
    d_rhs[-1] += lam * T0

    # Thomas Algorithm timing
    start = tm.time()
    _ = thomas_algorithm(a_diag, b_diag, c_diag, d_rhs)
    thomas_time = tm.time() - start
    thomas_times.append(thomas_time)

    # Gauss Elimination timing
    A = np.diag(b_diag) + np.diag(a_diag, k=-1) + np.diag(c_diag, k=1)
    start = tm.time()
    _ = np.linalg.solve(A, d_rhs)
    gauss_time = tm.time() - start
    gauss_times.append(gauss_time)

# Plot computation times
plt.figure(figsize=(10, 6))
plt.plot(grid_sizes, thomas_times, label='Thomas Algorithm', marker='o')
plt.plot(grid_sizes, gauss_times, label='Gauss Elimination', marker='s')
plt.xlabel('Number of Internal Grid Points')
plt.ylabel('Computation Time (s)')
plt.title('Computation Time vs Grid Size')
plt.legend()
plt.grid(True)
plt.show()
