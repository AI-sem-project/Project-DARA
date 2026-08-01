"""
Preprocessing.py
-----------------
Preprocessing module for Project DARA (Design and Analysis of Regression
Algorithms).
"""

import csv
import numpy as np

# --------------------------------------------------------------------
# District codes, used only to print friendly names in menus.
# --------------------------------------------------------------------
DISTRICT_NAMES = {
    "K": "Kathmandu",
    "L": "Lalitpur",
    "B": "Bhaktapur",
}

# Every dataset file Project DARA produced. Train/test ratio is NOT
# fixed here -- it is supplied by the caller (main.py asks the user).
DATASET_FILES = {
    "raw_1":      "Datasets_for_DARA_1.csv",
    "outliers_1": "Datasets_for_DARA_1_outliers.csv",
    "ghost_1":    "Datasets_for_DARA_1_ghost.csv",
    "final_1":    "Datasets_for_DARA_1_final.csv",
    "raw_2":      "Datasets_for_DARA_2.csv",
    "outliers_2": "Datasets_for_DARA_2_outliers.csv",
    "ghost_2":    "Datasets_for_DARA_2_ghost.csv",
    "final_2":    "Datasets_for_DARA_2_final.csv",
}


def clean_text(text):
    """
    Strips whitespace and stray leading/trailing quote characters from
    a raw text field.
    """
    return text.strip().strip("'\"").strip()


def load_dataset_rows(file_path):
    """
    Reads one Project DARA CSV file and returns parsed row dictionaries
    plus the detected column schema.
    """
    rows = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        schema = "floor" if "Floor" in fieldnames else "no_floor"

        for raw_row in reader:
            if not raw_row or not raw_row.get("Location"):
                continue  # Skip blank trailing lines

            row = {}
            for col in fieldnames:
                value = raw_row[col]
                if col in ("Location", "District", "Area", "Facing"):
                    row[col] = clean_text(value)
                else:
                    row[col] = float(value)
            rows.append(row)

    return rows, schema, fieldnames


def split_indices(n, train_ratio):
    """
    Shuffles row indices 0..n-1 and partitions them into train/test
    index arrays according to `train_ratio`.
    """
    indices = np.random.permutation(n)
    cutoff = int(round(train_ratio * n))
    return indices[:cutoff], indices[cutoff:]


def write_split_csv(file_path, fieldnames, rows):
    """Writes a list of row-dicts to a CSV file using the original column order."""
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_dataset(file_path, train_ratio, output_dir=".", split_label="dataset"):
    """
    Runs the full preprocessing pipeline on a single Project DARA CSV file.
    """
    np.random.seed(42)

    rows, schema, fieldnames = load_dataset_rows(file_path)
    n = len(rows)
    train_idx, test_idx = split_indices(n, train_ratio)

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    train_csv_path = f"{output_dir}/{split_label}_train.csv"
    test_csv_path = f"{output_dir}/{split_label}_test.csv"
    write_split_csv(train_csv_path, fieldnames, train_rows)
    write_split_csv(test_csv_path, fieldnames, test_rows)

    if schema == "floor":
        feature_names = [
            "District_PriceCode", "Area_PriceCode", "Land Area",
            "Bedrooms", "Floor", "Road Access", "Facing_PriceCode",
        ]
    else:
        feature_names = [
            "District_PriceCode", "Area_PriceCode", "Land Area",
            "Bedrooms", "Road Access", "Facing_PriceCode",
        ]

    def row_to_features(row):
        return [row[col] for col in feature_names]

    X_all = np.array([row_to_features(r) for r in rows], dtype=np.float64)
    y_all = np.array([r["Price"] for r in rows], dtype=np.float64)

    X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    X_min = X_train_raw.min(axis=0)
    X_max = X_train_raw.max(axis=0)
    range_diff = X_max - X_min
    range_diff = np.where(range_diff == 0, 1.0, range_diff)

    X_train = (X_train_raw - X_min) / range_diff
    X_test = (X_test_raw - X_min) / range_diff

    global_mean_price = float(np.mean(y_train))

    district_lookup = {}
    area_lookup = {}
    facing_lookup = {}
    area_district_map = {}

    for r in train_rows:
        district_lookup.setdefault(r["District"], r["District_PriceCode"])
        area_lookup.setdefault(r["Area"], r["Area_PriceCode"])
        facing_lookup.setdefault(r["Facing"], r["Facing_PriceCode"])
        area_district_map.setdefault(r["Area"], r["District"])

    areas_by_district = {}
    for area, district in area_district_map.items():
        areas_by_district.setdefault(district, []).append(area)
    for district in areas_by_district:
        areas_by_district[district].sort()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "schema": schema,
        "X_min": X_min,
        "X_max": X_max,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_csv_path": train_csv_path,
        "test_csv_path": test_csv_path,
        "district_lookup": district_lookup,
        "area_lookup": area_lookup,
        "facing_lookup": facing_lookup,
        "area_district_map": area_district_map,
        "global_mean_price": global_mean_price,
        "districts_available": sorted(district_lookup.keys()),
        "areas_by_district": areas_by_district,
        "facings_available": sorted(facing_lookup.keys()),
    }


def run_preprocessing_pipeline(train_ratios, base_path=".", output_dir="."):
    """
    Runs the full Project DARA preprocessing pipeline on all EIGHT dataset files.
    """
    results = {}
    for label, filename in DATASET_FILES.items():
        path = filename if base_path in (".", "") else f"{base_path}/{filename}"
        ratio = train_ratios[label]
        results[label] = process_dataset(
            path, ratio, output_dir=output_dir, split_label=label
        )
    return results


if __name__ == "__main__":
    print("--- Running Project DARA Preprocessing Pipeline on all 8 datasets ---\n")
    demo_ratios = {label: 0.70 for label in DATASET_FILES}
    all_data = run_preprocessing_pipeline(demo_ratios)

    for label, ds in all_data.items():
        print(f"[{label}]  schema={ds['schema']}  "
              f"X_train={ds['X_train'].shape}  X_test={ds['X_test'].shape}")
        print(f"    Features: {ds['feature_names']}")
        print(f"    Train CSV: {ds['train_csv_path']}")
        print(f"    Test CSV : {ds['test_csv_path']}")
        print(f"    Global mean price (train): {ds['global_mean_price']:,.2f}\n")

    print("Processing finished. Every dataset's matrices are ready for the regression engines.")