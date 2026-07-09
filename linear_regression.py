"""Linear regression engines using the Normal Equation (matrix math only)."""

import numpy as np


# ============================================================
# ENGINE 1: Simple Linear Regression (1 feature)
# theta = (X^T X)^-1 X^T y
# ============================================================

def fit_simple(x, y):
    """Fit a simple linear regression model using the normal equation.

    Args:
        x: 1D array-like of a single feature's values.
        y: 1D array-like of target values.

    Returns:
        A 1D numpy array [theta_0, theta_1] representing the
        intercept and slope.
    """
    x = np.array(x, dtype=float).reshape(-1, 1)
    y = np.array(y, dtype=float).reshape(-1, 1)

    design_matrix = np.hstack([np.ones((x.shape[0], 1)), x])

    transpose = design_matrix.T
    theta = np.linalg.inv(transpose @ design_matrix) @ transpose @ y

    return theta.flatten()


def predict_simple(x, theta):
    """Generate predictions from a fitted simple linear regression model.

    Args:
        x: 1D array-like of a single feature's values.
        theta: Fitted coefficient vector [theta_0, theta_1].

    Returns:
        A numpy array of predicted values.
    """
    x = np.array(x, dtype=float).reshape(-1, 1)
    design_matrix = np.hstack([np.ones((x.shape[0], 1)), x])

    return design_matrix @ theta








