"""
ridge_regression.py

Ridge Regression implementation for Project DARA
(Design and Analysis of Regression Algorithms).

This module implements Ridge Regression using the closed-form Normal
Equation with L2 regularization. It operates directly on the
pre-processed, pre-encoded numeric input features described in the
Project DARA proposal (rooms, flats, land area, location index, road
access, facing direction) and predicts a single continuous target
(house price).

Design constraints (per Project DARA specification):
    * Native Python lists are used for all data handling and business
      logic.
    * NumPy is used strictly for matrix algebra (constructing arrays,
      transposing, building the identity matrix, matrix multiplication,
      and computing the pseudo-inverse). It is never used for feature
      generation, looping, or statistics.
    * No third-party ML/scientific libraries (scikit-learn, scipy,
      etc.) are used. Everything is implemented from first principles.
"""

from typing import List, Optional

import numpy as np


class RidgeRegression:
    """Ridge Regression solved via the closed-form Normal Equation.

    Mathematical Background:
        Ridge Regression extends Ordinary Least Squares by adding an
        L2 penalty on the coefficients to reduce overfitting and
        stabilize the solution when features are correlated or the
        design matrix is ill-conditioned.

        Objective:
            min_theta  ||y - X * theta||^2 + lambda * ||theta||^2

        Closed-form solution:
            theta = (X^T X + lambda * I)^(-1) X^T y

        The intercept (bias) term is excluded from regularization, as
        is standard practice, because penalizing it would bias the
        model's baseline prediction toward zero.

    Attributes:
        lambda_ (float): L2 regularization strength.
        coefficients (Optional[List[float]]): Learned parameters,
            including the intercept as the first entry. None until
            `fit()` has been called.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """Initializes the Ridge Regression model.

        Args:
            lambda_:
                Regularization strength. Must be >= 0. A value of 0
                reduces Ridge Regression to Ordinary Least Squares.

        Raises:
            ValueError: If `lambda_` is negative.
        """
        if lambda_ < 0:
            raise ValueError("lambda_ must be >= 0.")

        self.lambda_: float = lambda_
        self.coefficients: Optional[List[float]] = None
        self._num_features: Optional[int] = None

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """Fits the Ridge Regression model using the training data.

        Mathematical Explanation:
            Builds the design matrix by prepending an intercept
            column of ones to X, then solves the regularized Normal
            Equation:

                theta = (X^T X + lambda * I')^(-1) X^T y

            where I' is the identity matrix with the entry
            corresponding to the intercept set to zero, so the
            intercept is not penalized.

        Args:
            X:
                Training features as a list of rows, where each row
                is a list of numeric feature values.
            y:
                Training targets as a list of numeric values, one per
                row in X.

        Returns:
            None

        Raises:
            ValueError: If X or y are empty, if their lengths do not
                match, or if rows of X have inconsistent length.
        """
        self._validate_fit_input(X, y)

        self._num_features = len(X[0])
        design_matrix = self._add_intercept(X)

        design_matrix_np = np.array(design_matrix, dtype=float)
        targets_np = np.array(y, dtype=float).reshape(-1, 1)

        num_columns = design_matrix_np.shape[1]
        regularization_matrix = np.identity(num_columns)
        regularization_matrix[0, 0] = 0.0  # Do not regularize the intercept.

        transpose_design = design_matrix_np.T
        gram_matrix = transpose_design @ design_matrix_np
        penalized_matrix = gram_matrix + self.lambda_ * regularization_matrix

        theta = np.linalg.pinv(penalized_matrix) @ transpose_design @ targets_np

        self.coefficients = theta.flatten().tolist()

    def predict(self, X: List[List[float]]) -> List[float]:
        """Predicts target values for the given input features.

        Mathematical Explanation:
            Applies the learned linear model:

                y_hat = theta_0 + theta_1 * x_1 + ... + theta_n * x_n

            by prepending the intercept column to X and computing the
            dot product with the learned coefficient vector.

        Args:
            X:
                Input features as a list of rows, where each row is a
                list of numeric feature values.

        Returns:
            A list of predicted target values, one per row in X.

        Raises:
            RuntimeError: If called before `fit()`.
            ValueError: If the number of features in X does not match
                the number of features seen during training.
        """
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before calling predict().")

        self._validate_predict_input(X)

        design_matrix = self._add_intercept(X)
        design_matrix_np = np.array(design_matrix, dtype=float)
        theta_np = np.array(self.coefficients, dtype=float).reshape(-1, 1)

        predictions_np = design_matrix_np @ theta_np
        return predictions_np.flatten().tolist()

    @staticmethod
    def _add_intercept(X: List[List[float]]) -> List[List[float]]:
        """Prepends a column of ones to the feature matrix.

        Args:
            X:
                Input features as a list of rows.

        Returns:
            A new list of rows, each prefixed with a leading 1.0 to
            represent the intercept term.
        """
        return [[1.0] + [float(value) for value in row] for row in X]

    @staticmethod
    def _validate_fit_input(X: List[List[float]], y: List[float]) -> None:
        """Validates training data shape and consistency.

        Args:
            X:
                Training features.
            y:
                Training targets.

        Returns:
            None

        Raises:
            ValueError: If X or y are empty, lengths mismatch, or rows
                of X are inconsistent in length.
        """
        if not X or not y:
            raise ValueError("X and y must not be empty.")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples "
                f"(got {len(X)} and {len(y)})."
            )

        row_length = len(X[0])
        if row_length == 0:
            raise ValueError("Rows of X must contain at least one feature.")

        for row in X:
            if len(row) != row_length:
                raise ValueError("All rows in X must have the same length.")

    def _validate_predict_input(self, X: List[List[float]]) -> None:
        """Validates prediction input shape against training data.

        Args:
            X:
                Input features to validate.

        Returns:
            None

        Raises:
            ValueError: If X is empty, rows are inconsistent in
                length, or the feature count does not match the
                training data.
        """
        if not X:
            raise ValueError("X must not be empty.")

        row_length = len(X[0])
        for row in X:
            if len(row) != row_length:
                raise ValueError("All rows in X must have the same length.")

        if row_length != self._num_features:
            raise ValueError(
                f"Expected {self._num_features} features, got {row_length}."
            )