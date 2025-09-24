import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# -----------------------------
# 1. CONSTANTS & HYPERPARAMETERS
# -----------------------------
REPORT_DIR = "report_data"
LEARNING_RATE = 0.05
MAX_ITERATIONS = 2000
TOLERANCE = 1e-6
ALPHA = 1.0
SAMPLES = 200
FEATURES = 10
NOISE = 0.5
RANDOM_SEED = 42

# -----------------------------
# 2. DATA GENERATION
# -----------------------------
def generate_synthetic_data(n_samples=SAMPLES, n_features=FEATURES, noise=NOISE, random_state=RANDOM_SEED):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    
    # First half of features are relevant, the rest are irrelevant.
    true_weights = np.array([3.0, -1.5, 2.0, 0.8, -2.2] + [0.0] * (n_features - 5))
    
    y = X @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights
# -----------------------------
# 4. GRADIENT DESCENT BASE CLASS (FOR CODE REUSE)
# -----------------------------
class GradientDescentRegressor:
    def _init_(self, learning_rate=LEARNING_RATE, max_iter=MAX_ITERATIONS, tol=TOLERANCE):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def fit(self, X, y):
        """Template method for model training."""
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(self, X):
        """Makes predictions on new data."""
        return X @ self.weights + self.bias
# -----------------------------
# 5. REGRESSION MODELS
# -----------------------------
class LinearRegressionGD(GradientDescentRegressor):
    """Linear Regression model without regularization."""
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            error = y_pred - y

            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            cost = mean_squared_error(y, y_pred)
            self.cost_history.append(cost)

            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Linear Regression converged at iteration {i}")
                break

class RidgeRegressionGD(GradientDescentRegressor):
    """Ridge Regression model with L2 regularization."""
    def __init__(self, alpha=ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            error = y_pred - y
            
            dw = (1 / n_samples) * X.T @ error + (2 * self.alpha * self.weights)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            cost = mean_squared_error(y, y_pred) + self.alpha * np.sum(self.weights ** 2)
            self.cost_history.append(cost)

            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Ridge Regression converged at iteration {i}")
                break

class LassoRegressionGD(GradientDescentRegressor):
    """Lasso Regression model with L1 regularization."""
    def __init__(self, alpha=ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            error = y_pred - y

            dw_mse = (1 / n_samples) * X.T @ error
            dw_l1 = self.alpha * np.sign(self.weights)
            dw = dw_mse + dw_l1

            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            cost = mean_squared_error(y, y_pred) + self.alpha * np.sum(np.abs(self.weights))
            self.cost_history.append(cost)

            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Lasso Regression converged at iteration {i}")
                break
