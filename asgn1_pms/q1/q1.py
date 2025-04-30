# Author: Nabayan Saha, 22CH3FP19
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
    
    def get_columns(self):
        return self.df[['boiling point (K)', 'critical temperature (K)', 'molweight', 'acentric factor']]

class LinearRegressionModel:
    def __init__(self, X, y):
        self.X = X.reshape(-1, 1)
        self.y = y
        self.model = LinearRegression()
    
    def train(self):
        self.model.fit(self.X, self.y)
    
    def predict(self):
        return self.model.predict(self.X)
    
    def evaluate(self):
        y_pred = self.predict()
        return r2_score(self.y, y_pred)
    
    def plot(self):
        y_pred = self.predict()
        plt.scatter(self.X, self.y, color="blue", label="Actual Data", alpha=0.5)
        plt.plot(self.X, y_pred, color="red", label="Linear Fit", linewidth=2)
        plt.xlabel("Molecular Weight")
        plt.ylabel("Boiling Point (K)")
        plt.title("Boiling Point vs Molecular Weight")
        plt.legend()
        plt.show()
    
    def summary(self):
        X_ols = sm.add_constant(self.X)
        ols_model = sm.OLS(self.y, X_ols).fit()
        return ols_model.summary()

class NormalEquation:
    @staticmethod
    def compute(X, y):
        return np.linalg.inv(X.T @ X) @ X.T @ y

class NeuralNetwork:
    def __init__(self, input_data, target_data, test_size=0.9):
        X_train, X_temp, y_train, y_temp = train_test_split(input_data, target_data, test_size=test_size, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        self.X_train, self.y_train = X_train, y_train
        self.model = self.build_model()
    
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(2,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['r2_score'])
        return model
    
    def train(self, epochs=100, batch_size=32):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, 
                                      validation_data=(self.X_val, self.y_val), verbose=1)
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.y_test, y_pred)
    
    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Model Training Loss')
        plt.legend()
        plt.show()

def main():
    data_loader = DataLoader('/kaggle/input/boiling-point-with-chemical-properties/data_file.csv')
    df = data_loader.get_columns()
    
    # Part (a)
    lin_reg_model = LinearRegressionModel(df['molweight'].values, df['boiling point (K)'].values)
    lin_reg_model.train()
    lin_reg_model.plot()
    print(f"R2 Score: {lin_reg_model.evaluate()}")
    print(lin_reg_model.summary())
    
    # Part (b)
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), 100, replace=False)
    sample_data = df.iloc[sample_indices]
    X = np.ones((100, 3))
    X[:, 1] = sample_data['molweight'].values
    X[:, 2] = sample_data['acentric factor'].values
    y = (sample_data['boiling point (K)'] / sample_data['critical temperature (K)']).values.reshape(-1, 1)
    theta = NormalEquation.compute(X, y)
    print(f"Theta values: {theta.flatten()}")
    
    # Part (c)
    scaler = StandardScaler()
    input_data = scaler.fit_transform(df[["molweight", "acentric factor"]])
    target_data = (df["boiling point (K)"] / df["critical temperature (K)"]).values 
    nn_model = NeuralNetwork(input_data, target_data)
    nn_model.train()
    nn_model.plot_loss()
    print(f"Neural Network R2 Score: {nn_model.evaluate()}")
    
    # Test Size vs R2 Score Analysis
    test_sizes = np.linspace(0.1, 0.9, 10)
    r2_scores = []
    for test_size in test_sizes:
        X_train, X_temp, y_train, y_temp = train_test_split(input_data, target_data, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        model = NeuralNetwork(X_train, y_train, test_size=test_size)
        model.train()
        r2_scores.append(model.evaluate())
    
    plt.scatter(test_sizes, r2_scores, color='b')
    plt.xlabel('Test Size')
    plt.ylabel('R^2 Score')
    plt.title('Test Size vs. R^2 Score')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()