"""
main.py
-------
The Orchestration & Evaluation module for Project DARA.

This script ties together the preprocessing pipeline and the four custom
regression engines. It handles data routing, type conversions, model 
training, predictions, and calculates evaluation metrics from scratch.
"""

# Import the Preprocessing pipeline
from Preprocessing import run_preprocessing_pipeline

# Import the four Regression Engines
from linear_regression import SimpleLinearRegression
from multiple_regression import MultipleLinearRegression
from Ridge_Regression import RidgeRegression
from Polynomial_Regression import PolynomialRegression


# ==========================================
# EVALUATION METRICS (Built from scratch)
# ==========================================

def calculate_mse(y_true: list[float], y_pred: list[float]) -> float:
    """Calculates Mean Squared Error."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def calculate_rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Calculates Root Mean Squared Error."""
    return calculate_mse(y_true, y_pred) ** 0.5

def calculate_r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """Calculates the R-squared (Coefficient of Determination)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    
    mean_y = sum(y_true) / n
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def print_metrics(model_name: str, y_true: list[float], y_pred: list[float]) -> None:
    """Helper function to print formatted evaluation metrics."""
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r_squared(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    # Formatted to show NPR clearly since Price was left unscaled
    print(f"RMSE:      NPR {rmse:,.2f}") 
    print(f"R-squared: {r2:.4f}")
    print("-" * 40)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Define file paths (Update these if your CSV names differ)
    DATASET_1 = "Datasets_for_DARA_1.csv"
    DATASET_2 = "Datasets_for_DARA_2.csv"
    
    print("Initializing Project DARA Orchestration...")
    print("Running Preprocessing Pipeline...\n")
    
    # 2. Run the preprocessor
    data = run_preprocessing_pipeline(
        DATASET_1, DATASET_2, train_ratio_1=0.70, train_ratio_2=0.80
    )
    
    # 3. Format Data for the Engines (Fixing the Type Mismatch Issue)
    # We will use Dataset 1 for this evaluation demonstration.
    
    # Extract matrices and convert to standard Python lists
    X_train_multi = data['X1_train'].tolist()
    X_test_multi = data['X1_test'].tolist()
    y_train = data['y1_train'].tolist()
    y_test = data['y1_test'].tolist()
    
    # Fix for Simple Linear Regression: 
    # Extract ONLY column index 1 ("Land Area") to act as our single feature
    x_train_simple = data['X1_train'][:, 1].tolist()
    x_test_simple = data['X1_test'][:, 1].tolist()


    # 4. Instantiate, Fit, Predict, and Evaluate All Models

    print("Training models and calculating metrics on Test Set (Dataset 1)...\n")
    print("=" * 40)

    # --- MODEL 1: Simple Linear Regression ---
    slr_model = SimpleLinearRegression()
    slr_model.fit(x_train_simple, y_train)
    slr_predictions = slr_model.predict(x_test_simple)
    print_metrics("Simple Linear Regression (Feature: Land Area)", y_test, slr_predictions)

    # --- MODEL 2: Multiple Linear Regression ---
    mlr_model = MultipleLinearRegression()
    mlr_model.fit(X_train_multi, y_train)
    mlr_predictions = mlr_model.predict(X_test_multi)
    print_metrics("Multiple Linear Regression (All Features)", y_test, mlr_predictions)

    # --- MODEL 3: Ridge Regression ---
    # Using lambda_ = 1.0 (You can tune this hyperparameter later)
    ridge_model = RidgeRegression(lambda_=1.0)
    ridge_model.fit(X_train_multi, y_train)
    ridge_predictions = ridge_model.predict(X_test_multi)
    print_metrics("Ridge Regression (Lambda = 1.0)", y_test, ridge_predictions)

    # --- MODEL 4: Polynomial Regression ---
    # Using degree = 2 for multivariate polynomial expansion
    poly_model = PolynomialRegression(degree=2)
    poly_model.fit(X_train_multi, y_train)
    poly_predictions = poly_model.predict(X_test_multi)
    print_metrics("Polynomial Regression (Degree = 2)", y_test, poly_predictions)

    print("Project DARA Pipeline Execution Complete.")