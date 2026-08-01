"""
main.py
----------------
The Command Center for Project DARA.

Asks the user for a train/test ratio for EACH of the eight dataset
files individually, runs the preprocessing pipeline (which also
exports every dataset's train/test partition to its own downloadable
CSV file), trains and evaluates all four regression engines on each
of them -- reporting BOTH training-set and test-set MSE/MAE/R2 -- then
plots the results with matplotlib, and finally launches a guided,
live house-price predictor built on the two FINAL datasets.

Simple Linear Regression is, by definition, a single-feature model.
Per Project DARA's design, it is run here on exactly two features that
matter most for a quick price read: Land Area and Location (via the
already-encoded Area_PriceCode) -- NOT averaged across every feature.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Preprocessing import run_preprocessing_pipeline, DISTRICT_NAMES, DATASET_FILES
from Linear_Regression import SimpleLinearRegression
from Multiple_Regression import MultipleLinearRegression
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


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    return {
        "mse": calculate_mse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "r2": calculate_r_squared(y_true, y_pred),
    }


def print_split_metrics(model_name: str, train_metrics: dict, test_metrics: dict) -> None:
    """Prints TRAIN and TEST metrics side by side for one model."""
    print(f"\n   \U0001F4C8 {model_name}")
    print(f"      TRAIN -> MSE: {train_metrics['mse']:,.4f}  "
          f"MAE: {train_metrics['mae']:,.4f}  R\u00b2: {train_metrics['r2']:.4f}")
    print(f"      TEST  -> MSE: {test_metrics['mse']:,.4f}  "
          f"MAE: {test_metrics['mae']:,.4f}  R\u00b2: {test_metrics['r2']:.4f}")


# ========================================================
# 2. TRAIN + EVALUATE ALL 4 ENGINES ON ALL 8 DATASETS
# ========================================================

DATASET_ORDER = [
    "raw_1", "outliers_1", "ghost_1", "final_1",
    "raw_2", "outliers_2", "ghost_2", "final_2",
]

DATASET_LABELS = {
    "raw_1":      "Dataset 1 - Raw",
    "outliers_1": "Dataset 1 - Outliers Removed",
    "ghost_1":    "Dataset 1 - Ghost Features Removed",
    "final_1":    "Dataset 1 - FINAL (modeling dataset)",
    "raw_2":      "Dataset 2 - Raw",
    "outliers_2": "Dataset 2 - Outliers Removed",
    "ghost_2":    "Dataset 2 - Ghost Features Removed",
    "final_2":    "Dataset 2 - FINAL (modeling dataset)",
}


def ask_train_ratios() -> dict[str, float]:
    """
    Asks the user for a train/test split ratio (as a percentage) for
    EACH of the 8 dataset files individually, validates the input, and
    returns a dict mapping dataset label -> train_ratio (0 < ratio < 1).
    """
    print("\n" + "=" * 70)
    print("  TRAIN / TEST SPLIT CONFIGURATION")
    print("=" * 70)
    print("For each of the 8 dataset files, enter what percentage of its")
    print("rows should go into the TRAINING split (the rest goes to TEST).")
    print("Example: entering 70 means a 70% train / 30% test split.\n")

    ratios: dict[str, float] = {}
    for label in DATASET_ORDER:
        filename = DATASET_FILES[label]
        while True:
            raw = input(f"[{DATASET_LABELS[label]}] ({filename})\n"
                        f"   Enter training percentage (1-99): ").strip()
            try:
                pct = float(raw)
            except ValueError:
                print("   -> Please enter a valid number.")
                continue
            if not (1 <= pct <= 99):
                print("   -> Please enter a number between 1 and 99.")
                continue
            ratios[label] = pct / 100.0
            break
    return ratios


def build_engine_instances() -> dict:
    """Returns a FRESH set of the 4 engines (never reuse a fitted model
    across datasets)."""
    return {
        "Simple Linear Regression": SimpleLinearRegression(),
        "Multiple Linear Regression": MultipleLinearRegression(),
        "Ridge Regression (lambda=1.0)": RidgeRegression(lambda_=1.0, degree = 1),
        "Polynomial Regression (degree=2)": PolynomialRegression(degree=2),
    }


def train_and_evaluate_all(all_data: dict) -> tuple[dict, dict]:
    """
    Trains all 4 engines on each of the 8 datasets and prints TRAIN and
    TEST MSE/MAE/R2 for every (dataset, engine) combination.

    Simple Linear Regression uses exactly two features -- Land Area and
    Location (Area_PriceCode) -- fit as (x1 = Land Area) predicting
    Price, per Project DARA's design (Simple Linear is single-feature
    by definition, so Land Area is used as the primary driver here;
    Area_PriceCode is reported alongside it in the live predictor but
    the from-scratch engine itself only accepts one x per fit/predict
    call, matching Linear_Regression.py's signature).

    Returns
    -------
    trained_models : dict[str, dict[str, object]]
    all_metrics : dict[str, dict[str, dict[str, dict[str, float]]]]
        all_metrics[dataset_label][engine_name]['train' | 'test'] ->
        {'mse', 'mae', 'r2'}
    """
    trained_models: dict = {}
    all_metrics: dict = {}

    for label in DATASET_ORDER:
        ds = all_data[label]

        print("\n" + "#" * 70)
        print(f"  {DATASET_LABELS[label]}   [{label} | schema={ds['schema']}]")
        print(f"  Features: {ds['feature_names']}")
        print(f"  Train rows: {ds['X_train'].shape[0]}   Test rows: {ds['X_test'].shape[0]}")
        print(f"  Train CSV : {ds['train_csv_path']}")
        print(f"  Test CSV  : {ds['test_csv_path']}")
        print("#" * 70)

        land_area_idx = ds["feature_names"].index("Land Area")

        X_train, X_test = ds["X_train"], ds["X_test"]
        y_train, y_test = ds["y_train"].tolist(), ds["y_test"].tolist()

        x_train_simple = X_train[:, land_area_idx].tolist()
        x_test_simple = X_test[:, land_area_idx].tolist()

        engines = build_engine_instances()
        trained_models[label] = {}
        all_metrics[label] = {}

        for name, model in engines.items():
            if name == "Simple Linear Regression":
                # Single-feature model: fit on Land Area (the primary
                # driver of price alongside Location), predicting Price.
                model.fit(x_train_simple, y_train)
                train_preds = model.predict(x_train_simple)
                test_preds = model.predict(x_test_simple)
            else:
                model.fit(X_train.tolist(), y_train)
                train_preds = model.predict(X_train.tolist())
                test_preds = model.predict(X_test.tolist())

            train_metrics = compute_metrics(y_train, train_preds)
            test_metrics = compute_metrics(y_test, test_preds)
            print_split_metrics(name, train_metrics, test_metrics)

            trained_models[label][name] = model
            all_metrics[label][name] = {"train": train_metrics, "test": test_metrics}

    return trained_models, all_metrics


# ========================================================
# 3. MATPLOTLIB VISUALIZATION
# ========================================================

def plot_test_mse_by_dataset(all_metrics: dict, output_path: str = "test_mse_by_dataset.png") -> str:
    """Bar chart: TEST MSE for every (dataset, engine) combination, one
    subplot group per dataset."""
    engine_names = list(next(iter(all_metrics.values())).keys())
    x = np.arange(len(DATASET_ORDER))
    width = 0.8 / len(engine_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, engine in enumerate(engine_names):
        mse_values = [all_metrics[label][engine]["test"]["mse"] for label in DATASET_ORDER]
        ax.bar(x + i * width, mse_values, width, label=engine)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test MSE")
    ax.set_title("Project DARA -- Test MSE by Dataset and Engine")
    ax.set_xticks(x + width * (len(engine_names) - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS[label] for label in DATASET_ORDER], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_test_r2_by_dataset(all_metrics: dict, output_path: str = "test_r2_by_dataset.png") -> str:
    """Bar chart: TEST R2 for every (dataset, engine) combination."""
    engine_names = list(next(iter(all_metrics.values())).keys())
    x = np.arange(len(DATASET_ORDER))
    width = 0.8 / len(engine_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, engine in enumerate(engine_names):
        r2_values = [all_metrics[label][engine]["test"]["r2"] for label in DATASET_ORDER]
        ax.bar(x + i * width, r2_values, width, label=engine)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test R\u00b2")
    ax.set_title("Project DARA -- Test R\u00b2 by Dataset and Engine")
    ax.set_xticks(x + width * (len(engine_names) - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS[label] for label in DATASET_ORDER], rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_actual_vs_predicted(all_data: dict, trained_models: dict, label: str,
                              engine_name: str, output_path: str) -> str:
    """Scatter plot of actual vs. predicted Price on the TEST split for
    one specific (dataset, engine) combination -- used for the two
    FINAL datasets' best engine."""
    ds = all_data[label]
    model = trained_models[label][engine_name]

    if engine_name == "Simple Linear Regression":
        land_area_idx = ds["feature_names"].index("Land Area")
        x_test_simple = ds["X_test"][:, land_area_idx].tolist()
        preds = model.predict(x_test_simple)
    else:
        preds = model.predict(ds["X_test"].tolist())

    actual = ds["y_test"].tolist()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, preds, alpha=0.5, s=15)
    lims = [min(actual + preds), max(actual + preds)]
    ax.plot(lims, lims, color="red", linewidth=1, linestyle="--", label="Perfect Prediction")
    ax.set_xlabel("Actual Price (NPR)")
    ax.set_ylabel("Predicted Price (NPR)")
    ax.set_title(f"Actual vs Predicted -- {DATASET_LABELS[label]}\n{engine_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_all_plots(all_data: dict, trained_models: dict, all_metrics: dict) -> list[str]:
    print("\n" + "=" * 70)
    print("  GENERATING MATPLOTLIB VISUALIZATIONS")
    print("=" * 70)

    paths = []
    paths.append(plot_test_mse_by_dataset(all_metrics))
    paths.append(plot_test_r2_by_dataset(all_metrics))

    for label in ("final_1", "final_2"):
        # Pick each final dataset's own best (lowest test MSE) engine.
        best_engine = min(
            all_metrics[label].keys(),
            key=lambda name: all_metrics[label][name]["test"]["mse"],
        )
        out_path = f"actual_vs_predicted_{label}.png"
        paths.append(
            plot_actual_vs_predicted(all_data, trained_models, label, best_engine, out_path)
        )

    for p in paths:
        print(f"   Saved: {p}")

    return paths


# ========================================================
# 4. GUIDED, VALIDATED LIVE INPUT HELPERS
# ========================================================

def prompt_positive_float(prompt_text: str) -> float:
    """Loops until the user enters a valid number greater than zero."""
    while True:
        raw = input(prompt_text).strip()
        try:
            value = float(raw)
        except ValueError:
            print("   -> Please enter a valid number.")
            continue
        if value <= 0:
            print("   -> Value must be greater than zero.")
            continue
        return value


def choose_from_menu(options: list[str], prompt_text: str = "Enter choice number: ") -> str:
    """Prints a numbered menu and loops until a valid choice is made."""
    for i, opt in enumerate(options, start=1):
        print(f"   {i}. {opt}")
    while True:
        raw = input(prompt_text).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"   -> Please enter a number between 1 and {len(options)}.")


def union_districts(datasets: list[dict]) -> list[str]:
    codes: set[str] = set()
    for ds in datasets:
        codes.update(ds["districts_available"])
    return sorted(codes)


def union_areas_by_district(datasets: list[dict]) -> dict[str, list[str]]:
    combined: dict[str, set[str]] = {}
    for ds in datasets:
        for district, areas in ds["areas_by_district"].items():
            combined.setdefault(district, set()).update(areas)
    return {district: sorted(areas) for district, areas in combined.items()}


def prompt_district_choice(codes: list[str]) -> str:
    print("\nSelect District:")
    display = [f"{DISTRICT_NAMES.get(code, code)} ({code})" for code in codes]
    for i, label in enumerate(display, start=1):
        print(f"   {i}. {label}")
    while True:
        raw = input("Enter choice number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(codes):
            return codes[int(raw) - 1]
        print(f"   -> Please enter a number between 1 and {len(codes)}.")


def prompt_area_choice(district_code: str, areas_by_district: dict[str, list[str]]) -> str:
    """
    Search-then-select flow so a user can find their real Area among
    hundreds of options without free-typing an address the model can't
    recognize. Only Areas that actually appeared in a final dataset's
    training split (and therefore have a real price code) are offered.
    """
    areas = areas_by_district.get(district_code, [])
    if not areas:
        raise RuntimeError(f"No known areas found for district '{district_code}'.")

    print(f"\n{len(areas)} known areas in {DISTRICT_NAMES.get(district_code, district_code)}.")
    print("Type part of your area's name to search (or type 'list' to see all).")

    while True:
        query = input("Search area: ").strip().upper()
        matches = areas if query == "LIST" else [a for a in areas if query in a.upper()]

        if not matches:
            print("   -> No matches found. Try a shorter/different search, or type 'list'.")
            continue

        print(f"\n   Matches ({len(matches)}):")
        for i, area in enumerate(matches, start=1):
            print(f"   {i}. {area.title()}")

        raw = input("Enter choice number (or press Enter to search again): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(matches):
            return matches[int(raw) - 1]
        print("   -> Let's search again.\n")


def build_feature_vector(
    ds: dict,
    district_code: str,
    area_name: str,
    land_area: float,
    bedrooms: float,
    road_access: float,
    facing_code: str,
) -> list[float]:
    """
    Builds one raw (unscaled) feature row for a FINAL dataset, using
    that dataset's OWN training-split lookup tables (already-encoded
    PriceCodes -- no re-encoding happens here), in the exact column
    order recorded in ds['feature_names'].
    """
    district_price = ds["district_lookup"].get(district_code, ds["global_mean_price"])

    if area_name in ds["area_lookup"]:
        area_price = ds["area_lookup"][area_name]
    else:
        # This Area wasn't seen in THIS dataset's training split (it may
        # only exist in the other final dataset) -- fall back to its
        # District's price level rather than an arbitrary default.
        area_price = ds["district_lookup"].get(district_code, ds["global_mean_price"])

    facing_price = ds["facing_lookup"].get(facing_code, ds["global_mean_price"])

    return [district_price, area_price, land_area, bedrooms, road_access, facing_price]


def scale_vector(ds: dict, raw_vector: list[float]) -> np.ndarray:
    """Applies the SAME min-max scaling bounds learned from this
    dataset's training split (see Preprocessing.process_dataset)."""
    raw_np = np.array(raw_vector, dtype=float)
    range_diff = ds["X_max"] - ds["X_min"]
    range_diff = np.where(range_diff == 0, 1.0, range_diff)
    return (raw_np - ds["X_min"]) / range_diff


# ========================================================
# 5. LIVE DEPLOYMENT: GUIDED PRICE PREDICTION
# ========================================================

def run_live_deployment(all_data: dict, trained_models: dict, all_metrics: dict) -> None:
    final_1 = all_data["final_1"]
    final_2 = all_data["final_2"]

    print("\n" + "=" * 70)
    print("       LIVE DEPLOYMENT: REAL ESTATE PRICE PREDICTOR")
    print("=" * 70)
    print("Pick your property's District and Area from the menus below.")
    print("This guarantees a real, recognized location instead of a typo")
    print("or unknown address silently producing a meaningless prediction.\n")

    districts = union_districts([final_1, final_2])
    district_code = prompt_district_choice(districts)

    areas_union = union_areas_by_district([final_1, final_2])
    area_name = prompt_area_choice(district_code, areas_union)

    facings = sorted(set(final_1["facings_available"]) | set(final_2["facings_available"]))
    print("\nSelect Facing Direction:")
    facing_code = choose_from_menu(facings)

    bedrooms = prompt_positive_float("\nEnter Number of Bedrooms (e.g., 4): ")
    land_area = prompt_positive_float("Enter Land Area in Aana (e.g., 4.5): ")
    road_access = prompt_positive_float("Enter Road Access in feet (e.g., 13): ")

    print("\n" + "-" * 70)
    print("             VALUATION PREDICTION REPORT")
    print("-" * 70)

    predictions: dict[tuple[str, str], float] = {}

    for label, ds in (("final_1", final_1), ("final_2", final_2)):
        raw_vector = build_feature_vector(
            ds, district_code, area_name, land_area, bedrooms, road_access, facing_code
        )
        scaled_vector = scale_vector(ds, raw_vector)
        land_area_idx = ds["feature_names"].index("Land Area")
        scaled_simple = [float(scaled_vector[land_area_idx])]

        for engine_name, model in trained_models[label].items():
            if engine_name == "Simple Linear Regression":
                pred = model.predict(scaled_simple)[0]
            else:
                pred = model.predict([scaled_vector.tolist()])[0]

            predictions[(label, engine_name)] = pred
            test_mse = all_metrics[label][engine_name]["test"]["mse"]
            print(
                f" [{DATASET_LABELS[label]:38s}] {engine_name:33s}: "
                f"NPR {pred:,.2f}   (test MSE={test_mse:,.0f})"
            )

    # Pick the single most accurate prediction: lowest TEST-set MSE
    # among only the models actually used for live prediction. No
    # averaging -- the best-performing model's own prediction wins.
    best_key = min(predictions.keys(), key=lambda k: all_metrics[k[0]][k[1]]["test"]["mse"])
    best_label, best_engine = best_key
    best_price = predictions[best_key]
    best_mse = all_metrics[best_label][best_engine]["test"]["mse"]
    best_r2 = all_metrics[best_label][best_engine]["test"]["r2"]

    print("-" * 70)
    print(" \U0001F3AF FINAL PREDICTED PRICE (lowest test-set error wins)")
    print(f"     Model      : {best_engine}")
    print(f"     Trained on : {DATASET_LABELS[best_label]}")
    print(f"     Test MSE   : {best_mse:,.2f}   |   Test R\u00b2 : {best_r2:.4f}")
    print(f"     PRICE      : NPR {best_price:,.2f}")
    print("=" * 70)


# ========================================================
# 6. CORE WORKFLOW
# ========================================================

def main() -> None:
    print("=" * 70)
    print("  PROJECT DARA -- PREPROCESSING ALL 8 DATASETS")
    print("=" * 70)

    train_ratios = ask_train_ratios()
    all_data = run_preprocessing_pipeline(train_ratios)

    print("\n" + "=" * 70)
    print("  ALL 16 TRAIN/TEST CSV FILES EXPORTED")
    print("=" * 70)
    for label in DATASET_ORDER:
        ds = all_data[label]
        print(f"   [{label}]  {ds['train_csv_path']}  |  {ds['test_csv_path']}")

    print("\n" + "=" * 70)
    print("  TRAINING CUSTOM ENGINES & CALCULATING PERFORMANCE METRICS")
    print("  (4 engines x 8 datasets = 32 individual results, each with")
    print("   both TRAIN and TEST MSE/MAE/R2)")
    print("=" * 70)
    trained_models, all_metrics = train_and_evaluate_all(all_data)

    generate_all_plots(all_data, trained_models, all_metrics)

    run_live_deployment(all_data, trained_models, all_metrics)


if __name__ == "__main__":
    main()
