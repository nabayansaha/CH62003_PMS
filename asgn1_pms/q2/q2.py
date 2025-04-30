# Author: Nabayan Saha, 22CH3FP19
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class AntoineEquation:
    @staticmethod
    def pressure(T, A, B, C):
        return 10**(A - B / (T + C))

class Component:
    def __init__(self, name, coeff_ranges):
        self.name = name
        self.coeff_ranges = coeff_ranges
    
    def get_coefficients(self, T):
        def interpolate_coefficients(T, ranges):
            for (T_min, T_max), coeffs in ranges:
                if T_min <= T <= T_max:
                    return coeffs
            
            sorted_ranges = sorted(ranges, key=lambda x: (x[0][0] + x[0][1]) / 2)
            
            for i in range(len(sorted_ranges) - 1):
                range1, coeffs1 = sorted_ranges[i]
                range2, coeffs2 = sorted_ranges[i + 1]
                mid1 = (range1[0] + range1[1]) / 2
                mid2 = (range2[0] + range2[1]) / 2
                
                if mid1 <= T <= mid2:
                    w2 = (T - mid1) / (mid2 - mid1)
                    w1 = 1 - w2
                    return tuple(c1 * w1 + c2 * w2 for c1, c2 in zip(coeffs1, coeffs2))
            
            return sorted_ranges[0][1] if T < sorted_ranges[0][0][0] else sorted_ranges[-1][1]
        
        return interpolate_coefficients(T, self.coeff_ranges)

class VLECalculator:
    def __init__(self, components, Ptotal):
        self.components = components
        self.Ptotal = Ptotal
    
    def calculate_K(self, T):
        K_values = []
        for component in self.components:
            A, B, C = component.get_coefficients(T)
            P = AntoineEquation.pressure(T, A, B, C)
            K_values.append(P / self.Ptotal)
        return K_values
    
    @staticmethod
    def equation_to_solve(V, T, F, z, K):
        return sum((z[i] * F) / ((K[i] - 1) * V + F) for i in range(len(z))) - 1
    
    def solve_VLE(self, T, F, z):
        K = self.calculate_K(T)
        V = fsolve(self.equation_to_solve, F / 2, args=(T, F, z, K))[0]
        x = [(z[i] * F) / ((K[i] - 1) * V + F) for i in range(len(z))]
        return T, K[0], x[0], V

def main():
    benzene = Component("Benzene", [
        ((333.4, 373.5), (4.72583, 1660.652, -1.461)),
        ((297.9, 318.0), (0.14591, 39.165, -261.236)),
        ((421.56, 554.8), (4.60362, 1701.073, 20.806)),
        ((287.70, 354.07), (4.01814, 1203.835, -53.226))
    ]) # source: https://webbook.nist.gov/cgi/inchi?ID=C71432&Mask=4&Plot=on&Type=ANTOINE
    
    toluene = Component("Toluene", [
        ((273.13, 297.89), (4.23679, 1426.448, -45.957)),
        ((303.0, 343.0), (4.08245, 1346.382, -53.508)),
        ((420.00, 580.00), (4.54436, 1738.123, 0.394)),
        ((308.52, 384.66), (4.07827, 1343.943, -53.773)),
        ((273.0, 323.0), (4.14157, 1377.578, -50.507))
    ]) # source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4&Type=ANTOINE&Plot=on
    
    oxylene = Component("O-Xylene", [
        ((336.61, 418.52), (4.12928, 1478.244, -59.076)),
        ((273.0, 323.0), (4.93755, 1901.373, -26.268))
    ]) # source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C95476&Mask=4&Type=ANTOINE&Plot=on
    
    components = [benzene, toluene, oxylene]
    Ptotal = 1.01325
    F = 100
    z = [0.6, 0.25, 0.15]
    T_range = np.arange(373, 394, 10)
    
    vle_calculator = VLECalculator(components, Ptotal)
    results = [vle_calculator.solve_VLE(T, F, z) for T in T_range]
    
    T_vals, K_vals, x_vals, V_vals = zip(*results)
    V_K_x_vals = [V * k * x for V, k, x in zip(V_vals, K_vals, x_vals)]
    
    max_index = np.argmax(V_K_x_vals)
    max_T = T_vals[max_index]
    max_value = V_K_x_vals[max_index]

    # Plot
    plt.plot(T_vals, V_K_x_vals, label="Benzene in Vapor Phase")
    plt.axvline(x=max_T, linestyle="dotted", color="red", label=f"Max at {max_T} K")
    plt.scatter([max_T], [max_value], color="red") 
    plt.annotate(f"Max: {max_value:.2f}", (max_T, max_value), textcoords="offset points", xytext=(10, -10), ha="center")

    plt.xlabel('Temperature (K)')
    plt.ylabel('Benzene Amount in Vapor (Kmol)')
    plt.title('Benzene amount vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
