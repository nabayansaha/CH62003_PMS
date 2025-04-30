import numpy as np
import matplotlib.pyplot as plt

L = 0.01 
T0 = 300 
alpha = 1.8e-4  
a = 443.42
b = 0.04675
omega = 1
dx = 0.0025  # Spatial step (in meters)
n_steps = 100
dt = 1/n_steps  # Time step (in seconds)
n_cycles = 5  # Number of cycles

def k(T):
    return a - b*T

n_grid = int(L / dx) + 1
lambda_ = alpha * dt / (dx**2)
lambda_ = 0.7
if lambda_ > 0.5:
    print("Warning: The stability condition is not satisfied (lambda > 0.5).")
print(f"Stability condition (lambda): {lambda_}")
time = np.arange(0, n_cycles * 1 / omega, 1 / n_steps)
T = np.full((n_grid, len(time)), T0)  
T[0, :] = T0 * (1 + np.sin(2 * np.pi * omega * time))  
T[:, 0] = T0
T[- 1, :] = T0 
for n in range(0, len(time) - 1):
    for i in range(1, n_grid - 1):
        T[i, n + 1] = T[i, n] + lambda_ * (
            (1 - (b * T[i, n] / a)) * (T[i + 1, n] - 2 * T[i, n] + T[i - 1, n])
            - (b / (4 * a)) * (T[i + 1, n] - T[i - 1, n]) ** 2)

x_vals = [0.25 * L, 0.5 * L, 0.75 * L, L]
x_indices = [int(x / dx) for x in x_vals]

plt.figure(figsize=(10, 6))
for idx in x_indices:
    plt.plot(time, T[idx, :], label=f'x = {x_vals[x_indices.index(idx)]} m')

plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title(f'Temperature Profiles for Explicit Finite Difference Scheme (lambda = {lambda_}) violating stability')
plt.legend()
plt.grid(True)
plt.show()
