# Author: Nabayan Saha, 22CH3FP19
import numpy as np
from scipy.optimize import fsolve

class DistillationSolver:
    def __init__(self, components, T_values, V_values, L_values, U_values, A, Z, P_ref=101325):
        self.components = components
        self.T_values = np.array(T_values)
        self.V_values = np.array(V_values)
        self.L_values = np.array(L_values)
        self.U_values = np.array(U_values)
        self.A = np.array(A)
        self.Z = Z
        self.P_ref = P_ref
        
    def vapor_pressure(self, T, C1, C2, C3, C4, C5):
        return np.exp(C1 + C2 / T + C3 * np.log(T) + C4 * (T ** C5))
    
    def compute_ki_matrix(self):
        Ki_matrix = np.zeros((len(self.components), len(self.T_values)))
        for j, T in enumerate(self.T_values):
            for i, (comp, params) in enumerate(self.components.items()):
                Ki_matrix[i, j] = self.vapor_pressure(T, *params) / self.P_ref
        return Ki_matrix
    
    def compute_cij_matrix(self, Ki_matrix):
        rows, cols = Ki_matrix.shape
        Cij_matrix = np.zeros((rows, cols - 1))
        for j in range(cols - 1):
            for i in range(rows):
                Cij_matrix[i, j] = -self.V_values[j + 1] * Ki_matrix[i, j + 1]
        return Cij_matrix
    
    def compute_bij_matrix(self, Ki_matrix):
        rows, cols = Ki_matrix.shape
        Bij_matrix = np.zeros((rows, cols))
        for j in range(cols):
            for i in range(rows):
                Bij_matrix[i, j] = self.V_values[j] * Ki_matrix[i, j] + self.L_values[j] + self.U_values[j]
        return Bij_matrix
    
    def construct_tdm(self, Bij_matrix, Cij_matrix):
        TDM_matrices = {}
        num_stages = Bij_matrix.shape[1]
        for i, comp in enumerate(self.components.keys()):
            TDM = np.zeros((num_stages, num_stages))
            for j in range(num_stages):
                TDM[j, j] = Bij_matrix[i, j]
                if j < num_stages - 1:
                    TDM[j, j + 1] = Cij_matrix[i, j]
                if j > 0:
                    TDM[j, j - 1] = self.A[j]
            TDM_matrices[comp] = TDM
        return TDM_matrices
    
    def solve_tdm_systems(self, TDM_matrices):
        solutions = {}
        for comp, TDM in TDM_matrices.items():
            try:
                X = np.linalg.solve(TDM, self.Z[comp])
                solutions[comp] = X
            except np.linalg.LinAlgError:
                print(f"Matrix for {comp} is singular and cannot be solved.")
        return solutions
    
    def normalize_solutions(self, solutions):
        iterations = 100
        tol = 1e-6
        for _ in range(iterations):
            total = np.sum(list(solutions.values()), axis=0)
            if np.all(np.abs(total - 1) < tol):
                break
            solutions = {comp: sol / total for comp, sol in solutions.items()}
        return solutions
    
    def solve_temperatures(self, solutions):
        def equation(T, j):
            return sum(self.vapor_pressure(T, *self.components[comp]) / self.P_ref * solutions[comp][j] for comp in self.components) - 1
        
        return [fsolve(equation, self.T_values[j], args=(j,))[0] for j in range(len(self.T_values))]
    
    def run(self):
        while True:
            Ki_matrix = self.compute_ki_matrix()
            Cij_matrix = self.compute_cij_matrix(Ki_matrix)
            Bij_matrix = self.compute_bij_matrix(Ki_matrix)
            TDM_matrices = self.construct_tdm(Bij_matrix, Cij_matrix)
            solutions = self.solve_tdm_systems(TDM_matrices)
            solutions = self.normalize_solutions(solutions)
            T_final = self.solve_temperatures(solutions)
            
            error = sum(((np.array(T_final) - self.T_values) / np.array(T_final))**2)
            print("Error:", error)
            if error < 1e-6:
                break
            self.T_values = np.array(T_final)
        
        return solutions, self.T_values

if __name__ == "__main__":
    components = {
        "Methanol":  [81.768, -6876, -8.7078, 7.1926E-6, 2],
        "Ethanol":   [74.475, -7164.3, -7.327, 3.1340E-6, 2],
        "Propanol":  [88.134, -8498.6, -9.0766, 8.3303E-18, 6]
    }
    
    T_values = [338, 346.25, 354.5, 362.75, 371]
    V_values = [0, 2600, 2600, 2600, 2600]
    L_values = [2000, 2000, 3000, 3000, 400]
    U_values = [600, 0, 0, 0, 0]
    A = [0, -2000, -2000, -3000, -3000]
    Z = {
        "Methanol": np.array([0, 0, 600, 0, 0]),
        "Ethanol": np.array([0, 0, 200, 0, 0]),
        "Propanol": np.array([0, 0, 200, 0, 0])
    }
    
    solver = DistillationSolver(components, T_values, V_values, L_values, U_values, A, Z)
    solutions, final_temperatures = solver.run()
    print("Final Solutions:", solutions)
    print("Final Temperatures:", final_temperatures)