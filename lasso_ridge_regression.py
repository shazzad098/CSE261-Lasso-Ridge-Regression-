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