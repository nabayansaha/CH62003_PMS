import numpy as np
from scipy.optimize import fsolve
import math

class DistillationColumn:
    def __init__(self, feed_composition, distillate_composition, bottoms_composition, K_values):
        self.feed_composition = feed_composition
        self.distillate_composition = distillate_composition
        self.bottoms_composition = bottoms_composition
        self.K_values = K_values
        self.alpha = self.calculate_relative_volatility(K_values["iC5"], K_values["nC4"])
    
    def calculate_relative_volatility(self, K_HK, K_LK):
        return K_HK / K_LK

    def fenske_min_stages(self):
        xD_HK = self.distillate_composition["iC5"] / sum(self.distillate_composition.values())
        xB_HK = self.bottoms_composition["iC5"] / sum(self.bottoms_composition.values())
        xD_LK = self.distillate_composition["nC4"] / sum(self.distillate_composition.values())
        xB_LK = self.bottoms_composition["nC4"] / sum(self.bottoms_composition.values())
        return np.log((xD_HK / xD_LK) * (xB_LK / xB_HK)) / np.log(self.alpha)

    def nonkey_distribution(self):
        nonkeys = set(self.feed_composition.keys()) - {"iC5", "nC4"}
        return {comp: (self.distillate_composition.get(comp, 0), self.bottoms_composition.get(comp, 0)) for comp in nonkeys}

    def underwood_theta(self, q=1.0):
        total_feed = sum(self.feed_composition.values())
        z = {comp: amount / total_feed for comp, amount in self.feed_composition.items()}
        components = list(self.K_values.keys())
        alpha_values = [self.K_values[comp] / self.K_values["nC4"] for comp in components]
        z_values = [z[comp] for comp in components]

        def equation(theta):
            return sum(alpha_i * z_i / (alpha_i - theta) for alpha_i, z_i in zip(alpha_values, z_values)) - (1 - q)
        
        return fsolve(equation, 1.1)[0]

    def calculate_R_min(self):
        theta = self.underwood_theta()
        total_dist = sum(self.distillate_composition.values())
        xD = [self.distillate_composition.get(comp, 0) / total_dist for comp in self.K_values.keys()]
        alpha_values = [self.K_values[comp] / self.K_values["nC4"] for comp in self.K_values.keys()]
        return sum(alpha_i * xd_i / (alpha_i - theta) for alpha_i, xd_i in zip(alpha_values, xD)) 

    # source: https://www.scielo.br/j/bjce/a/6GCR3ndMjBJSjxL4rTqN7Bt/
    def gilliland_stages(self, R_min, N_min):
        R = 1.3 * R_min
        x = (R - R_min) / (R + 1)
        y = 1 - np.exp(((1 + 54.4 * x) * (x - 1)) / ((11 + 117.2 * x) * np.sqrt(x)))
        return (N_min + 1) / (1 - y)

    def run(self):
        print(f"Estimated relative volatility (alpha): {self.alpha:.2f}")
        N_min = math.ceil(self.fenske_min_stages())
        print(f"Minimum number of stages (N_min): {N_min:.2f}")
        print("Nonkey component distributions:")
        for component, (dist, bot) in self.nonkey_distribution().items():
            print(f"{component}: Distillate = {dist}, Bottoms = {bot}")
        R_min = self.calculate_R_min()
        print(f"Minimum reflux ratio (R_min): {R_min:.2f}")
        N_actual = math.ceil(self.gilliland_stages(R_min, N_min))
        print(f"Actual number of stages required: {N_actual:.2f}")

if __name__ == "__main__":
    feed_composition = {"iC4": 12, "nC4": 448, "iC5": 36, "nC5": 15, "C6": 23, "C7": 39.1, "C8": 272.2, "C9": 31.0}
    distillate_composition = {"iC4": 12, "nC4": 442, "iC5": 13, "nC5": 1}
    bottoms_composition = {"nC4": 6, "iC5": 23, "nC5": 14, "C6": 23, "C7": 39.1, "C8": 272.2, "C9": 31.0}
    K_values = {"iC4": 3.0, "nC4": 2.5, "iC5": 1.2, "nC5": 1.0, "C6": 0.6, "C7": 0.25, "C8": 0.12, "C9": 0.1}# source: https://wiki.whitson.com/phase_behavior/properties/kvalues/
    
    column = DistillationColumn(feed_composition, distillate_composition, bottoms_composition, K_values)
    column.run()
