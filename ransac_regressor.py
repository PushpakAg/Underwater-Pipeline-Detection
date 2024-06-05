#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from copy import copy

class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X, y):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        return self

    def predict(self, X):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return np.dot(X, self.params)

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred)**2

def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

class RANSAC:
    def __init__(self, n, k, t, d, model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error):
        self.n = n  # Minimum number of data points to estimate parameters
        self.k = k  # Maximum iterations allowed
        self.t = t  # Threshold value for determining if points fit well
        self.d = d  # Number of close data points required to assert the model fits well
        self.model = model
        self.loss = loss
        self.metric = metric
        self.best_fit = None
        self.best_error = np.inf
        self.points = None
        self.fail = True

    def fit(self, X, y):
        rng = np.random.RandomState(123)
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0])
            maybe_inliers = ids[:self.n]
            test_points = ids[self.n:]
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])
            test_pred = maybe_model.predict(X[test_points])
            test_error = self.loss(y[test_points], test_pred)
            also_inliers = test_points[test_error < self.t]
            if len(also_inliers) > self.d:
                self.fail = False
                better_model_inliers = np.concatenate([maybe_inliers, also_inliers])
                better_model = copy(self.model).fit(X[better_model_inliers], y[better_model_inliers])
                better_model_error = self.metric(y[better_model_inliers], better_model.predict(X[better_model_inliers]))
                if better_model_error < self.best_error:
                    self.best_error = better_model_error
                    self.best_fit = better_model
                    self.points = better_model_inliers
        return self

    def predict(self, X):
        if not self.fail:
            return self.best_fit.predict(X)
        else:
            return None

# Generate synthetic data
np.random.seed(0)
n_samples = 100
n_outliers = 30

X = np.linspace(0, 10, n_samples)[:, np.newaxis]
y = 3 * X.squeeze() + np.random.normal(0, 2, n_samples)

# Add outliers
X[:n_outliers] = np.linspace(0, 10, n_outliers)[:, np.newaxis]
y[:n_outliers] = 50 + 10 * np.random.randn(n_outliers)

# Instantiate and use RANSAC
ransac = RANSAC(n=10, k=100, t=10, d=50, model=LinearRegressor())
ransac.fit(X, y)

# Plot
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

plt.scatter(X, y, color='yellowgreen', marker='.', label='Data')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', label='RANSAC regressor')
plt.legend()
plt.show()