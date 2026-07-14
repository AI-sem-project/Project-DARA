"""Multiple Linear Regression engine using the Normal Equation."""

from typing import List

import numpy as np


class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None
        self._num_features = None

    def fit(self, X: List[List[float | int]], y: List[float | int]) -> None:
        """Fits the Multiple Linear Regression model using training data.

        Mathematical Explanation:
            Builds the design matrix by prepending an intercept
            column of ones to X, then solves the Normal Equation:

                theta = (X^T X)^-1 X^T y

        Args:
            X:
                Training features as a list of rows, where each row
                is a list of numeric feature values.
            y:
                Training targets, one per row in X.

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

        transpose_design = design_matrix_np.T
        gram_matrix = transpose_design @ design_matrix_np

        theta = np.linalg.pinv(gram_matrix) @ transpose_design @ targets_np
        self.coefficients = theta.flatten().tolist()

    def predict(self, X: List[List[float | int]]) -> List[float | int]:
        """Applies the learned linear model to generate predictions.

        Mathematical Explanation:
            Applies the learned linear model:

                y_hat = theta_0 + theta_1 * x_1 + ... + theta_n * x_n

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
    def _add_intercept(X: List[List[float | int]]) -> List[List[float | int]]:
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
    def _validate_fit_input(
        X: List[List[float | int]], y: List[float | int]
    ) -> None:
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

    def _validate_predict_input(self, X: List[List[float | int]]) -> None:
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