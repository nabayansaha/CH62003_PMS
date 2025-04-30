import numpy as np
import matplotlib.pyplot as plt

L = 0.01
T0 = 300
a = 443.42
b = 0.04675
omega = 1
dx = 0.0025
dt = 0.01  # fixed for clarity
rho = 8000  # density (kg/m^3)
Cp = 500  # specific heat (J/kg·K)
n_steps = 100
n_cycles = 5
time = np.arange(0, n_cycles / omega, dt)
n_grid = int(L / dx) + 1

T = np.full((n_grid, len(time)), T0)
T[0, :] = T0 * (1 + np.sin(2 * np.pi * omega * time))
T[:, 0] = T0
T[-1, :] = T0

def k(T_val):
    return a - b * T_val

def thomas_algorithm(a_, b_, c_, d_):
    n = len(d_)
    # Create copies to avoid modifying the input arrays
    a_copy = np.copy(a_)
    b_copy = np.copy(b_)
    c_copy = np.copy(c_)
    d_copy = np.copy(d_)
    
    # Forward sweep
    for i in range(1, n):
        w = a_copy[i-1] / b_copy[i-1]
        b_copy[i] -= w * c_copy[i-1]
        d_copy[i] -= w * d_copy[i-1]
    
    # Backward substitution
    x = np.zeros(n)
    x[-1] = d_copy[-1] / b_copy[-1]
    for i in reversed(range(n-1)):
        x[i] = (d_copy[i] - c_copy[i] * x[i+1]) / b_copy[i]
    
    return x

for n in range(len(time) - 1):
    # Prepare TDM system
    num_internal_points = n_grid - 2  # Number of internal grid points
    a_diag = np.zeros(num_internal_points)  # sub-diagonal
    b_diag = np.zeros(num_internal_points)  # main diagonal
    c_diag = np.zeros(num_internal_points)  # super-diagonal
    d_rhs = np.zeros(num_internal_points)   # right-hand side
    
    for i in range(num_internal_points):
        grid_i = i + 1  # Convert to actual grid index (skip boundary at 0)
        k_val = k(T[grid_i, n])
        lam = k_val * dt / (rho * Cp * dx**2)
        
        # Fill in the matrix rows
        if i > 0:  # Not the first row
            a_diag[i] = -lam
        
        b_diag[i] = 1 + 2*lam
        
        if i < num_internal_points - 1:  # Not the last row
            c_diag[i] = -lam
        
        # RHS includes the current temperature and boundary conditions
        d_rhs[i] = T[grid_i, n]
        
        # Include boundary conditions in the first and last internal points
        if i == 0:  # First internal point (affected by left boundary)
            d_rhs[i] += lam * T[0, n+1]  # Left boundary at next time step
        
        if i == num_internal_points - 1:  # Last internal point (affected by right boundary)
            d_rhs[i] += lam * T0  # Right boundary (constant)
    
    # Print the tridiagonal matrix for the first time step
    if n == 0:
        print("Tridiagonal Matrix (A, B, C) and RHS (D) at t = 0:")
        for i in range(num_internal_points):
            a_val = a_diag[i] if i > 0 else 0
            c_val = c_diag[i] if i < num_internal_points - 1 else 0
            print(f"A={a_val:.4f}, B={b_diag[i]:.4f}, C={c_val:.4f}, D={d_rhs[i]:.4f}")
    
    # Solve the system - adjust arrays for Thomas algorithm
    T_internal = thomas_algorithm(
        a_=a_diag[1:],       # Sub-diagonal (starts from second row)
        b_=b_diag,           # Main diagonal
        c_=c_diag[:-1],      # Super-diagonal (ends at second-to-last row)
        d_=d_rhs             # RHS vector
    )
    
    # Update internal points
    T[1:-1, n+1] = T_internal
    # Boundary conditions are already set

# Plot results
x_vals = [0.25 * L, 0.5 * L, 0.75 * L, L]
x_indices = [int(x / dx) for x in x_vals]

plt.figure(figsize=(10, 6))
for i, idx in enumerate(x_indices):
    plt.plot(time, T[idx, :], label=f'x = {x_vals[i]:.4f} m')

plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profiles (Implicit Scheme)')
plt.legend()
plt.grid(True)
plt.show()