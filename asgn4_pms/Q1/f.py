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
# Re-run simulation for part (f) with the same setup
dx = 0.0025
n_grid = int(L / dx) + 1
T = np.full((n_grid, len(time)), T0)
T[0, :] = T0 * (1 + np.sin(2 * np.pi * omega * time))  # Left boundary
T[:, 0] = T0
T[-1, :] = T0

# Time-marching with Thomas Algorithm
for n in range(len(time) - 1):
    num_internal = n_grid - 2
    a_diag = np.zeros(num_internal - 1)
    b_diag = np.zeros(num_internal)
    c_diag = np.zeros(num_internal - 1)
    d_rhs = np.zeros(num_internal)
    
    for i in range(num_internal):
        grid_i = i + 1
        k_val = k(T[grid_i, n])
        lam = k_val * dt / (rho * Cp * dx**2)
        
        if i > 0:
            a_diag[i - 1] = -lam
        b_diag[i] = 1 + 2 * lam
        if i < num_internal - 1:
            c_diag[i] = -lam
        
        d_rhs[i] = T[grid_i, n]
        if i == 0:
            d_rhs[i] += lam * T[0, n+1]
        if i == num_internal - 1:
            d_rhs[i] += lam * T0
    
    T_internal = thomas_algorithm(a_diag, b_diag, c_diag, d_rhs)
    T[1:-1, n+1] = T_internal

# Calculate quantities of interest
T_left = T[0, :]  # T(x=0, t)
T_avg = np.mean(T, axis=0)  # Average across all x

# Plot T(x=0,t) and T_avg(t)
plt.figure(figsize=(10, 6))
plt.plot(time, T_left, label='T(x=0, t)', color='blue')
plt.plot(time, T_avg, label='Average Temperature', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Boundary Temperature vs Average Temperature')
plt.legend()
plt.grid(True)
plt.show()

omega_values = np.linspace(0.5, 100, 200)
amplitudes = []

for omega in omega_values:
    # Reinitialize temperature matrix
    time = np.arange(0, n_cycles / omega, dt)
    T = np.full((n_grid, len(time)), T0)
    T[0, :] = T0 * (1 + np.sin(2 * np.pi * omega * time))
    T[:, 0] = T0
    T[-1, :] = T0

    for n in range(len(time) - 1):
        num_internal = n_grid - 2
        a_diag = np.zeros(num_internal - 1)
        b_diag = np.zeros(num_internal)
        c_diag = np.zeros(num_internal - 1)
        d_rhs = np.zeros(num_internal)

        for i in range(num_internal):
            grid_i = i + 1
            k_val = k(T[grid_i, n])
            lam = k_val * dt / (rho * Cp * dx**2)

            if i > 0:
                a_diag[i - 1] = -lam
            b_diag[i] = 1 + 2 * lam
            if i < num_internal - 1:
                c_diag[i] = -lam

            d_rhs[i] = T[grid_i, n]
            if i == 0:
                d_rhs[i] += lam * T[0, n + 1]
            if i == num_internal - 1:
                d_rhs[i] += lam * T0

        T_internal = thomas_algorithm(a_diag, b_diag, c_diag, d_rhs)
        T[1:-1, n + 1] = T_internal

    # Analyze output at x = L
   # Analyze output at x = 0.75L
    idx_075L = int(0.75 * L / dx)
    T_075L = T[idx_075L, :]
    amp = (np.max(T_075L) - np.min(T_075L)) / 2
    amplitudes.append(amp)


# Plot amplitude vs ω
plt.figure(figsize=(10, 6))
plt.plot(omega_values, amplitudes, marker='o')
plt.xlabel('ω (rad/s)')
plt.ylabel('Amplitude at x = 0.75L')
plt.title('Response Amplitude at Far End vs Frequency')
plt.grid(True)
plt.show()

# Estimate critical ω: where amplitude falls below 1% of boundary oscillation
boundary_amp = T0  # From the sine boundary (300*(1 + sin(...)))
threshold = 0.01 * boundary_amp
critical_indices = [i for i, amp in enumerate(amplitudes) if amp < threshold]

if critical_indices:
    omega_c = omega_values[critical_indices[0]]
    print(f"Estimated critical ω = {omega_c:.2f} rad/s")
else:
    print("No critical ω found in scanned range.")
