import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 0.01
T0 = 300
a = 443.42
b = 0.04675
omega = 1
dx = 0.0025
dt = 0.01  # fixed for clarity
rho = 8000  # density (kg/m^3)
Cp = 500  # specific heat (J/kg·K)
n_cycles = 5
time = np.arange(0, n_cycles / omega, dt)
n_grid = int(L / dx) + 1

# Temperature matrix
T = np.zeros((n_grid, len(time)))
# Initialize with constant temperature
for i in range(n_grid):
    for j in range(len(time)):
        T[i, j] = T0

# Set boundary conditions
for j in range(len(time)):
    T[0, j] = T0 * (1 + np.sin(2 * np.pi * omega * time[j]))
    T[-1, j] = T0

def thermal_conductivity(temp):
    """Calculate thermal conductivity"""
    return a - b * temp

def solve_tridiagonal(a, b, c, d):
    """Solve a tridiagonal matrix system"""
    n = len(d)
    x = np.zeros(n)
    
    # Make copies of arrays
    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()
    d_copy = d.copy()
    
    # Forward elimination
    for i in range(1, n):
        m = a_copy[i] / b_copy[i-1]
        b_copy[i] -= m * c_copy[i-1]
        d_copy[i] -= m * d_copy[i-1]
    
    # Back substitution
    x[n-1] = d_copy[n-1] / b_copy[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_copy[i] - c_copy[i] * x[i+1]) / b_copy[i]
    
    return x

# Time stepping
for n in range(len(time) - 1):
    # Set up tridiagonal system for internal points
    n_internal = n_grid - 2
    a_diag = np.zeros(n_internal)  # lower diagonal
    b_diag = np.zeros(n_internal)  # main diagonal
    c_diag = np.zeros(n_internal)  # upper diagonal
    d_rhs = np.zeros(n_internal)   # right-hand side
    
    # Fill in the tridiagonal system
    for i in range(n_internal):
        grid_i = i + 1  # Actual grid index
        
        # Get temperature at current grid point
        temp = T[grid_i, n]
        
        # Calculate thermal conductivity and lambda
        k_val = thermal_conductivity(temp)
        lam = k_val * dt / (rho * Cp * dx**2)
        
        # Set up matrix coefficients
        if i > 0:
            a_diag[i] = -lam
            
        b_diag[i] = 1 + 2*lam
        
        if i < n_internal - 1:
            c_diag[i] = -lam
        
        # Right-hand side
        # lambda_i * T_{i-1}^n + (1-2*lambda_i) * T_i^n + lambda_i * T_{i+1}^n
        d_rhs[i] = lam * T[grid_i-1, n] + (1-2*lam) * T[grid_i, n] + lam * T[grid_i+1, n]
        
        # Handle boundary conditions
        if i == 0:  # First internal point
            # Contribution from left boundary T[0, n+1]
            a_diag[i] = 0  # No contribution from previous internal point
            d_rhs[i] += lam * T[0, n+1]  # Add effect of left boundary
            
        if i == n_internal - 1:  # Last internal point
            # Contribution from right boundary T[-1, n+1]
            c_diag[i] = 0  # No contribution from next internal point
            d_rhs[i] += lam * T[-1, n+1]  # Add effect of right boundary
    
    # Print the first few rows of the system at first time step
    if n == 0:
        print("Crank-Nicolson System at t = 0:")
        for i in range(min(5, n_internal)):
            a_val = a_diag[i] if i > 0 else 0
            c_val = c_diag[i] if i < n_internal - 1 else 0
            print(f"Row {i+1}: a={a_val:.4f}, b={b_diag[i]:.4f}, c={c_val:.4f}, d={d_rhs[i]:.4f}")
    
    solution = solve_tridiagonal(a_diag, b_diag, c_diag, d_rhs)
    for i in range(n_internal):
        T[i+1, n+1] = solution[i]

# Plot results
x_vals = [0.25 * L, 0.5 * L, 0.75 * L, L]
x_indices = [int(x / dx) for x in x_vals]

plt.figure(figsize=(10, 6))
for i, idx in enumerate(x_indices):
    plt.plot(time, T[idx, :], label=f'x = {x_vals[i]:.4f} m')

plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profiles (Crank-Nicolson Scheme)')
plt.legend()
plt.grid(True)
plt.show()