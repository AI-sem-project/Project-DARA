"""
orchestration.py
----------------
The Command Center for Project DARA.
Handles pipeline execution, metrics calculation from scratch, 
and live interactive deployment predictions.
"""

import numpy as np
# Import the Preprocessing pipeline
from Preprocessing import run_preprocessing_pipeline

# Import the four Regression Engines
from linear_regression import SimpleLinearRegression
from multiple_regression import MultipleLinearRegression
from Ridge_Regression import RidgeRegression
from Polynomial_Regression import PolynomialRegression


# ========================================================
# 1. EVALUATION METRICS ENGINE (Built Entirely From Scratch)
# ========================================================

def calculate_mse(y_true: list[float], y_pred: list[float]) -> float:
    """Calculates Mean Squared Error (MSE)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def calculate_mae(y_true: list[float], y_pred: list[float]) -> float:
    """Calculates Mean Absolute Error (MAE)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def calculate_r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """Calculates the Coefficient of Determination (R-Squared)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_y = sum(y_true) / n
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def print_metrics(model_name: str, y_true: list[float], y_pred: list[float]):
    """Helper to display metrics beautifully."""
    mse = calculate_mse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r_squared(y_true, y_pred)
    
    print(f"\n📈 Model: {model_name}")
    print(f"   - Mean Squared Error (MSE)      : {mse:,.4f}")
    print(f"   - Mean Absolute Error (MAE)     : {mae:,.4f}")
    print(f"   - Coefficient of Determination (R²): {r2:.4f}")


# ========================================================
# 2. CORE WORKFLOW & TRAINING LOOP
# ========================================================

def main():
    # FIXED: Passed the mandatory dataset filenames to the preprocessing pipeline
    dataset_1 = "Datasets_for_DARA_1.csv"
    dataset_2 = "Datasets_for_DARA_2.csv"
    data = run_preprocessing_pipeline(dataset_1, dataset_2)

    # Convert training/testing sets to standard native Python lists for compatibility
    y_train = data['y1_train'].tolist()
    y_test = data['y1_test'].tolist()

    X_train_multi = data['X1_train'].tolist()
    X_test_multi = data['X1_test'].tolist()

    # FIXED: Sliced index 1 (Land Area) instead of index 4 (Facing)
    x_train_simple = data['X1_train'][:, 1].tolist()
    x_test_simple = data['X1_test'][:, 1].tolist()

    print("=" * 60)
    print("  TRAINING CUSTOM ENGINES & CALCULATING PERFORMANCE METRICS")
    print("=" * 60)

    # --- Engine 1: Simple Linear Regression ---
    slr_model = SimpleLinearRegression()
    slr_model.fit(x_train_simple, y_train)
    slr_predictions = slr_model.predict(x_test_simple)
    print_metrics("Simple Linear Regression (Feature: Land Area)", y_test, slr_predictions)

    # --- Engine 2: Multiple Linear Regression ---
    mlr_model = MultipleLinearRegression()
    mlr_model.fit(X_train_multi, y_train)
    mlr_predictions = mlr_model.predict(X_test_multi)
    print_metrics("Multiple Linear Regression (All Features)", y_test, mlr_predictions)

    # --- Engine 3: Ridge Regression (L2 Regularization) ---
    ridge_model = RidgeRegression(lambda_=1.0)
    ridge_model.fit(X_train_multi, y_train)
    ridge_predictions = ridge_model.predict(X_test_multi)
    print_metrics("Ridge Regression (Lambda = 1.0)", y_test, ridge_predictions)

    # --- Engine 4: Polynomial Regression (Degree = 2) ---
    poly_model = PolynomialRegression(degree=2)
    poly_model.fit(X_train_multi, y_train)
    poly_predictions = poly_model.predict(X_test_multi)
    print_metrics("Polynomial Regression (Degree = 2)", y_test, poly_predictions)


    # ========================================================
    # 3. INTERACTIVE HOUSE PRICE PREDICTION SYSTEM
    # ========================================================
    print("\n" + "=" * 60)
    print("       LIVE DEPLOYMENT: REAL ESTATE PRICE PREDICTOR")
    print("=" * 60)
    print("Please enter the following 6 structural specifications:")

    # Gather user feature inputs
    ui_location     = input("1. Enter Location (e.g., Kathmandu, Lalitpur, Bhaktapur   : ").strip()
    ui_bedrooms     = float(input("2. Enter Number of Bedrooms (e.g., 4)               : "))
    ui_floors       = float(input("3. Enter Number of Floors (e.g., 2.5)               : "))
    ui_facing       = input("4. Enter House Facing Direction (e.g., North, West) : ").strip()
    ui_land_area    = float(input("5. Enter Land Area (in Aana)                        : "))
    ui_road_access  = float(input("6. Enter Road Access (in feet)                      : "))

    # --------------------------------------------------------
    # FEATURE TRANSFORMATION & NORMALIZATION (FIXED)
    # --------------------------------------------------------
    # Normalize categorical string cases to guarantee a safe dictionary match
    norm_loc = ui_location.upper()
    norm_facing = ui_facing.upper()

    # Turn raw strings into their unified numerical codes
    loc_code = float(data['location_map'].get(norm_loc, 0))
    facing_code = float(data['facing_map'].get(norm_facing, 0))

    # Construct the array preserving the project's layout sequence:
    # Column Order: [Location, Land Area, Bedrooms, Floor, Facing, Road Access]
    user_features_raw = np.array([
        loc_code,
        ui_land_area,
        ui_bedrooms,
        ui_floors,
        facing_code,
        ui_road_access
    ], dtype=float)

    # Map features precisely between 0 and 1 using parameters calculated from the training phase
    X1_min = data['X1_min']
    X1_max = data['X1_max']
    range_diff = X1_max - X1_min
    range_diff[range_diff == 0] = 1.0  # Prevent division by zero

    user_features_scaled_np = (user_features_raw - X1_min) / range_diff
    user_features_scaled = user_features_scaled_np.tolist()
    
    # Extract the isolated scaled Land Area (Index 1) for the simple engine
    user_land_area_scaled = user_features_scaled[1]

    # --------------------------------------------------------
    # COMPUTE MODEL PREDICTIONS
    # --------------------------------------------------------
    # 1. Simple Linear Engine (Evaluated exclusively on Land Area)
    pred_slr = slr_model.predict([user_land_area_scaled])[0]

    # 2. Multiple Linear Engine
    pred_mlr = mlr_model.predict([user_features_scaled])[0]

    # 3. Ridge Regression Engine
    pred_ridge = ridge_model.predict([user_features_scaled])[0]

    # 4. Polynomial Regression Engine
    pred_poly = poly_model.predict([user_features_scaled])[0]

    # Calculate the Final Combined Ensemble Average
    final_ensemble_price = (pred_slr + pred_mlr + pred_ridge + pred_poly) / 4

    # Display prediction report
    print("\n" + "-" * 50)
    print("             VALUATION PREDICTION REPORT")
    print("-" * 50)
    print(f" 🏠 Simple Linear Engine Price     : NPR {pred_slr:,.2f}")
    print(f" 🏢 Multiple Linear Engine Price   : NPR {pred_mlr:,.2f}")
    print(f" 🛡️  Ridge Regression Engine Price  : NPR {pred_ridge:,.2f}")
    print(f" 📐 Polynomial Regression Engine Price: NPR {pred_poly:,.2f}")
    print("-" * 50)
    print(f" 🎯 FINAL COMBINED PREDICTED PRICE  : NPR {final_ensemble_price:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
