"""Simple Linear Regression engine using the Normal Equation."""

from typing import List

import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, x: List[float | int], y: List[float | int]) -> None:
        """Fits the Simple Linear Regression model using training data.

        Mathematical Explanation:
            Builds the design matrix by prepending an intercept
            column of ones to x, then solves the Normal Equation:

                theta = (X^T X)^-1 X^T y

        Args:
            x:
                Training feature values (single feature).
            y:
                Training targets, one per value in x.

        Returns:
            None

        Raises:
            ValueError: If x or y are empty, or if their lengths do
                not match.
        """
        self._validate_fit_input(x, y)

        x_np = np.array(x, dtype=float).reshape(-1, 1)
        y_np = np.array(y, dtype=float).reshape(-1, 1)

        design_matrix = np.hstack([np.ones((x_np.shape[0], 1)), x_np])

        transpose_design = design_matrix.T
        gram_matrix = transpose_design @ design_matrix

        theta = np.linalg.inv(gram_matrix) @ transpose_design @ y_np
        self.coefficients = theta.flatten().tolist()

    def predict(self, x: List[float | int]) -> List[float | int]:
        """Applies the learned linear model to generate predictions.

        Mathematical Explanation:
            Applies the learned linear model:

                y_hat = theta_0 + theta_1 * x

        Args:
            x:
                Input feature values to predict on.

        Returns:
            A list of predicted target values, one per value in x.

        Raises:
            RuntimeError: If called before `fit()`.
        """
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before calling predict().")

        x_np = np.array(x, dtype=float).reshape(-1, 1)
        design_matrix = np.hstack([np.ones((x_np.shape[0], 1)), x_np])
        theta_np = np.array(self.coefficients, dtype=float).reshape(-1, 1)

        predictions_np = design_matrix @ theta_np
        return predictions_np.flatten().tolist()

    @staticmethod
    def _validate_fit_input(
        x: List[float | int], y: List[float | int]
    ) -> None:
        """Validates training data shape and consistency.

        Args:
            x:
                Training feature values.
            y:
                Training targets.

        Returns:
            None

        Raises:
            ValueError: If x or y are empty, or lengths mismatch.
        """
        if not x or not y:
            raise ValueError("x and y must not be empty.")

        if len(x) != len(y):
            raise ValueError(
                f"x and y must have the same number of samples "
                f"(got {len(x)} and {len(y)})."
            )