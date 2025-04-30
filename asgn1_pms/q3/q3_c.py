# Author: Nabayan Saha, 22CH3FP19
import numpy as np
from scipy.optimize import fsolve

class DistillationSolver:
    def __init__(self, components, T_values, P_ref=101325):
        self.components = components
        self.T_values = T_values
        self.P_ref = P_ref
        self.V_values = [0] + [2600] * (len(T_values) - 1)
        self.F = [0] * len(T_values)
        self.Z = {comp: np.zeros(len(T_values)) for comp in components}
    
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
    
    def compute_bij_matrix(self, Ki_matrix, L_values, U_values):
        rows, cols = Ki_matrix.shape
        Bij_matrix = np.zeros((rows, cols))
        for j in range(cols):
            for i in range(rows):
                Bij_matrix[i, j] = self.V_values[j] * Ki_matrix[i, j] + L_values[j] + U_values[j]
        return Bij_matrix
    
    def construct_tdm(self, A, Bij_matrix, Cij_matrix):
        TDM_matrices = {}
        num_stages = Bij_matrix.shape[1]
        for i, comp in enumerate(self.components.keys()):
            TDM = np.zeros((num_stages, num_stages))
            for j in range(num_stages):
                TDM[j, j] = Bij_matrix[i, j]
                if j < num_stages - 1:
                    TDM[j, j + 1] = Cij_matrix[i, j]
                if j > 0:
                    TDM[j, j - 1] = A[j]
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
    
    def find_optimal_feed_stage(self):
        min_error = float("inf")
        optimal_stage = None
        optimal_L_values = None
        
        for feed_stage in range(1, len(self.T_values) - 1):  
            L_values = [2000] * feed_stage + [3000] * (len(self.T_values) - 1 - feed_stage) + [400] 
            U_values = [600] + [0] * (len(self.T_values) - 1)
            A = np.array([0] + [-2000] * feed_stage + [-3000] * (len(self.T_values) - 1 - feed_stage) + [-400])
            
            self.F[feed_stage] = 1000
            self.Z = {comp: np.zeros(len(self.T_values)) for comp in self.components}
            self.Z["Methanol"][feed_stage] = 600
            self.Z["Ethanol"][feed_stage] = 200
            self.Z["Propanol"][feed_stage] = 200
            
            while True:
                Ki_matrix = self.compute_ki_matrix()
                Cij_matrix = self.compute_cij_matrix(Ki_matrix)
                Bij_matrix = self.compute_bij_matrix(Ki_matrix, L_values, U_values)
                TDM_matrices = self.construct_tdm(A, Bij_matrix, Cij_matrix)
                solutions = self.solve_tdm_systems(TDM_matrices)
                solutions = self.normalize_solutions(solutions)
                T_final = self.solve_temperatures(solutions)
                error = sum(((np.array(T_final) - np.array(self.T_values)) / np.array(T_final))**2)
                if error < 1e-6:
                    break
                self.T_values = T_final
            
            if error < min_error:
                min_error = error
                optimal_stage = feed_stage
                optimal_L_values = L_values.copy()
                final_solutions = solutions.copy()
                final_temperatures = T_final.copy()
        
        return optimal_stage, final_solutions, final_temperatures

def main():
    components = {
        "Methanol":  [81.768, -6876, -8.7078, 7.1926E-6, 2],
        "Ethanol":   [74.475, -7164.3, -7.327, 3.1340E-6, 2],
        "Propanol":  [88.134, -8498.6, -9.0766, 8.3303E-18, 6]
    }
    
    T_values = np.linspace(338, 400, 12)
    solver = DistillationSolver(components, T_values)
    optimal_stage, solutions, final_temperatures = solver.find_optimal_feed_stage()
    
    print("Optimal Feed Stage:", optimal_stage)
    print("Final Solutions:", solutions)
    print("Final Temperatures:", final_temperatures)
    
if __name__ == "__main__":
    main()