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

from typing import List, Optional, Tuple

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

    def __init__(self, lambda_: float = 1.0, degree: int = 1) -> None:
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
        if degree < 1:
            raise ValueError("degree must be >= 1.")


        self.lambda_: float = lambda_
        self.degree: int = degree
        self.coefficients: Optional[List[float]] = None
        self._num_features: Optional[int] = None
        self._exponent_table: Optional[List[Tuple[int, ...]]] = None

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
        self._exponent_table = self._build_exponent_table(
            self._num_features, self.degree
        )

        design_matrix = self._generate_polynomial_features(X)

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
        if self.coefficients is None or self._exponent_table is None:
            raise RuntimeError("Model must be fit before calling predict().")

        self._validate_predict_input(X)

        design_matrix = self._generate_polynomial_features(X)
        design_matrix_np = np.array(design_matrix, dtype=float)
        theta_np = np.array(self.coefficients, dtype=float).reshape(-1, 1)

        predictions_np = design_matrix_np @ theta_np
        return predictions_np.flatten().tolist()

    def _generate_polynomial_features(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Expands raw features into the full polynomial feature space.

        Mathematical Explanation:
            For each row x = (x_1, ..., x_n), and for each exponent
            tuple (e_1, ..., e_n) in the pre-built exponent table
            (covering every combination with 0 <= sum(e_i) <= degree),
            computes the monomial:

                x_1^(e_1) * x_2^(e_2) * ... * x_n^(e_n)

            The all-zero exponent tuple produces the constant "1"
            term, which serves as the intercept, so no separate
            intercept column needs to be added manually.

        Args:
            X:
                Input features as a list of rows.

        Returns:
            A list of rows, each containing the expanded polynomial
            features in a fixed, degree-ordered sequence.
        """
        expanded_rows: List[List[float]] = []
        for row in X:
            expanded_row = [
                self._evaluate_monomial(row, exponents)
                for exponents in self._exponent_table
            ]
            expanded_rows.append(expanded_row)
        return expanded_rows

    @staticmethod
    def _evaluate_monomial(
            row: List[float], exponents: Tuple[int, ...]
    ) -> float:
        """Evaluates a single monomial term for one data row.

        Args:
            row:
                A single sample's raw feature values.
            exponents:
                Exponent to apply to each corresponding feature.

        Returns:
            The product of each feature raised to its exponent.
        """
        value = 1.0
        for feature_value, exponent in zip(row, exponents):
            if exponent != 0:
                value *= float(feature_value) ** exponent
        return value

    @classmethod
    def _build_exponent_table(
            cls, num_features: int, degree: int
    ) -> List[Tuple[int, ...]]:
        """Builds the ordered list of exponent tuples for the expansion.

        Mathematical Explanation:
            Generates every exponent tuple (e_1, ..., e_n) such that
            0 <= sum(e_i) <= degree, ordered by increasing total
            degree. Within each total degree, tuples are generated by
            recursively assigning the largest possible exponent to
            the earliest feature first, which reproduces the standard
            expansion order (e.g. x^2, x*y, y^2 for two features at
            degree 2).

        Args:
            num_features:
                Number of raw input features.
            degree:
                Maximum total polynomial degree.

        Returns:
            A list of exponent tuples, one per generated polynomial
            term, ordered from the constant term up to the highest
            degree terms.
        """
        exponent_table: List[Tuple[int, ...]] = []
        for total_degree in range(0, degree + 1):
            exponent_table.extend(
                cls._tuples_with_fixed_sum(num_features, total_degree)
            )
        return exponent_table

    @classmethod
    def _tuples_with_fixed_sum(
            cls, num_slots: int, target_sum: int
    ) -> List[Tuple[int, ...]]:
        """Recursively generates exponent tuples summing to a fixed value.

        Args:
            num_slots:
                Number of features (positions) remaining to assign.
            target_sum:
                Total exponent sum that the tuple must add up to.

        Returns:
            A list of tuples of length `num_slots`, each summing
            exactly to `target_sum`, ordered from the largest first
            exponent to the smallest.
        """
        if num_slots == 1:
            return [(target_sum,)]

        tuples: List[Tuple[int, ...]] = []
        for first_exponent in range(target_sum, -1, -1):
            remaining_sum = target_sum - first_exponent
            for remainder in cls._tuples_with_fixed_sum(
                    num_slots - 1, remaining_sum
            ):
                tuples.append((first_exponent,) + remainder)
        return tuples

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