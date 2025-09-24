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
    
    true_weights = np.array([3.0, -1.5, 2.0, 0.8, -2.2] + [0.0] * (n_features - 5))
    
    y = X @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights

# -----------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# -----------------------------
# 4. GRADIENT DESCENT BASE CLASS (FOR CODE REUSE)
# -----------------------------
class GradientDescentRegressor:
    def __init__(self, learning_rate=LEARNING_RATE, max_iter=MAX_ITERATIONS, tol=TOLERANCE):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(self, X):
        return X @ self.weights + self.bias

# -----------------------------
# 5. REGRESSION MODELS
# -----------------------------
class LinearRegressionGD(GradientDescentRegressor):
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
            
            dw = (1 / n_samples) * X.T @ error + (self.alpha / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            cost = mean_squared_error(y, y_pred) + self.alpha * np.sum(self.weights ** 2)
            self.cost_history.append(cost)

            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Ridge Regression converged at iteration {i}")
                break

class LassoRegressionGD(GradientDescentRegressor):
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
            dw_l1[self.weights == 0] = self.alpha 
            
            dw = dw_mse + dw_l1
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            cost = mean_squared_error(y, y_pred) + self.alpha * np.sum(np.abs(self.weights))
            self.cost_history.append(cost)

            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Lasso Regression converged at iteration {i}")
                break

# -----------------------------
# 6. PLOTTING & REPORTING
# -----------------------------
def plot_coefficients(true_weights, models, feature_names):
    plt.figure(figsize=(12, 6))
   
    plt.plot(feature_names, true_weights, 'o-', label='True Coefficients', linewidth=2)
    plt.plot(feature_names, models['Linear'].weights, 's--', label='Linear Regression', alpha=0.8)
    plt.plot(feature_names, models['Ridge'].weights, '^--', label='Ridge Regression', alpha=0.8)
    plt.plot(feature_names, models['Lasso'].weights, 'd--', label='Lasso Regression', alpha=0.8)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Comparison of Regression Coefficients', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "coefficients_plot.png"), dpi=300)
    print("Coefficients plot saved.") 
    plt.show()

def plot_convergence(models):
    plt.figure(figsize=(10, 6))
    plt.plot(models['Linear'].cost_history, label='Linear Regression')
    plt.plot(models['Ridge'].cost_history, label='Ridge Regression')
    plt.plot(models['Lasso'].cost_history, label='Lasso Regression')
   
    plt.title('Convergence of Gradient Descent', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (Loss)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(REPORT_DIR, "convergence_plot.png"), dpi=300)
    print("Convergence plot saved.")
    plt.show()

def save_coefficients_to_csv(true_weights, models, feature_names):
    df = pd.DataFrame({
        'Feature': feature_names,
        'True': true_weights,
        'Linear': models['Linear'].weights,
        'Ridge': models['Ridge'].weights,
        'Lasso': models['Lasso'].weights
    })
   
    file_path = os.path.join(REPORT_DIR, "comparison_table.csv")
    df.to_csv(file_path, index=False)
    print(f"Coefficients saved to '{file_path}' for LaTeX report.")

def main():
    try:
        print("Starting Lasso vs Ridge Regression Comparison...\n")
       
        os.makedirs(REPORT_DIR, exist_ok=True)
       
        # 1. Generate data
        X, y, true_weights = generate_synthetic_data()
        feature_names = [f'X{i+1}' for i in range(X.shape[1])]
       
        # 2. Initialize and fit models
        lr = LinearRegressionGD(learning_rate=LEARNING_RATE, max_iter=MAX_ITERATIONS, tol=TOLERANCE)
        ridge = RidgeRegressionGD(learning_rate=LEARNING_RATE, max_iter=MAX_ITERATIONS, tol=TOLERANCE, alpha=ALPHA)
        lasso = LassoRegressionGD(learning_rate=LEARNING_RATE, max_iter=MAX_ITERATIONS, tol=TOLERANCE, alpha=ALPHA)

        models = {'Linear': lr, 'Ridge': ridge, 'Lasso': lasso}
       
        print("Fitting models...")
        for name, model in models.items():
            print(f"- Fitting {name} Regression...")
            model.fit(X, y)
       
        # 3. Evaluate models
        print("\nModel Performance:")
        y_pred_lr = lr.predict(X)
        y_pred_ridge = ridge.predict(X)
        y_pred_lasso = lasso.predict(X)

        print(f"  Linear Regression -> MSE: {mean_squared_error(y, y_pred_lr):.4f}, R²: {r2_score(y, y_pred_lr):.4f}")
        print(f"  Ridge Regression  -> MSE: {mean_squared_error(y, y_pred_ridge):.4f}, R²: {r2_score(y, y_pred_ridge):.4f}")
        print(f"  Lasso Regression  -> MSE: {mean_squared_error(y, y_pred_lasso):.4f}, R²: {r2_score(y, y_pred_lasso):.4f}")
       
        # 4. Generate plots and tables
        print("\nGenerating report assets...")
        plot_coefficients(true_weights, models, feature_names)
        plot_convergence(models)
        save_coefficients_to_csv(true_weights, models, feature_names)

        # 5. Key Observation
        print("\nKey Observation: Lasso's Sparsity")
        for i, coef in enumerate(lasso.weights):
            if abs(coef) < 0.01:
                print(f"  Feature {i+1} (X{i+1}) coefficient = {coef:.5f} → APPROACHING ZERO")
       
        print("\nConclusion: Lasso Regression is highly effective at inducing sparsity by driving irrelevant feature coefficients to zero. This makes it a powerful tool for feature selection, unlike Ridge Regression, which only shrinks coefficients without setting them to zero.")
        print("\nAll analysis and report assets saved in the 'report_data/' folder.")
       
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()