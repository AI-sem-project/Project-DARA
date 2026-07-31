"""
preprocessing.py
-----------------
Preprocessing module for Project DARA (Design and Analysis of Regression
Algorithms).

This module is the single gateway between the raw CSV files and every
regression algorithm in the project (Simple Linear, Multiple Linear,
Polynomial, Ridge). It processes all EIGHT Project DARA dataset files:

    Dataset 1 lineage: raw -> outliers-removed -> ghost-removed -> final
    Dataset 2 lineage: raw -> outliers-removed -> ghost-removed -> final

Two facts about the raw files drive most of the design below:

1. Two different column schemas exist across the 8 files:
       "floor" schema (raw, outliers files):
           Location, Land Area, Bedrooms, Floor, Facing, Road Access, Price
       "interaction" schema (ghost, final files):
           Location, Land Area, Bedrooms, Road Access,
           Land Area x Road Access, Facing, Price
   This module detects the schema from the CSV header, so both column
   layouts are handled by the same code path.

2. The `Location` column packs two pieces of information into one
   string, e.g. "'Imadol, L'" -> area = "Imadol", district = "L" where
   K = Kathmandu, L = Lalitpur, B = Bhaktapur. This module splits that
   into two separate features (District, Area) rather than treating the
   whole string as a single opaque category.

Encoding strategy -- PRICE-BASED (target) ENCODING
----------------------------------------------------
`Location` (split into District + Area) and `Facing` are no longer coded
with an arbitrary alphabetical index (which carries zero information
about price). Instead, each category is coded as a *smoothed average
Price* for that category:

    encoded_value = (n * category_mean_price + k * prior) / (n + k)

`n` is how many training rows fall in that category, `k` is a smoothing
constant, and `prior` is a fallback estimate used to pull rare
categories toward a more stable number instead of overfitting to a
handful of samples:
    - District's prior is the overall training-set mean price.
    - Area's prior is its own District's price code (hierarchical
      fallback -- an obscure Area still inherits a sensible estimate
      from the District it belongs to).
    - Facing's prior is the overall training-set mean price.

Critically, every one of these statistics is computed from the
TRAINING split only, then applied to both the training and test rows.
Computing them from the full dataset (train + test combined) before
splitting would leak the test set's own prices into a feature used to
predict price, silently inflating MSE/MAE/R2 on the test set. Fitting
on the training split only, and applying the learned lookup table to
the test split, keeps the evaluation honest.

No missing-value handling, duplicate removal, unit conversion, or
outlier/ghost-feature removal is performed here -- that cleaning was
already done manually across the different dataset files per the
Project DARA proposal. The only transformations here are: parsing
Location, price-based encoding of District/Area/Facing, min-max
scaling, shuffling, and train/test splitting.
"""

import csv
import re
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------
# District codes found in the raw Location strings, e.g. "Imadol, L'"
# --------------------------------------------------------------------
DISTRICT_NAMES: dict[str, str] = {
    "K": "Kathmandu",
    "L": "Lalitpur",
    "B": "Bhaktapur",
}

# Every dataset file Project DARA produced, and the train/test ratio the
# proposal specifies for its lineage (Dataset 1 -> 70/30, Dataset 2 -> 80/20).
DATASET_FILES: dict[str, tuple[str, float]] = {
    "raw_1":       ("Datasets_for_DARA_1.csv", 0.70),
    "outliers_1":  ("Datasets_for_DARA_1_outliers.csv", 0.70),
    "ghost_1":     ("Datasets_for_DARA_1_ghost.csv", 0.70),
    "final_1":     ("Datasets_for_DARA_1_final.csv", 0.70),
    "raw_2":       ("Datasets_for_DARA_2.csv", 0.80),
    "outliers_2":  ("Datasets_for_DARA_2_outliers.csv", 0.80),
    "ghost_2":     ("Datasets_for_DARA_2_ghost.csv", 0.80),
    "final_2":     ("Datasets_for_DARA_2_final.csv", 0.80),
}

# Smoothing constants for price-based encoding (see module docstring).
SMOOTHING_DISTRICT = 20.0
SMOOTHING_AREA = 10.0
SMOOTHING_FACING = 10.0


def normalize_category(text: str) -> str:
    """
    Normalizes a raw categorical string for consistent encoding.

    Strips surrounding whitespace and any stray leading/trailing quote
    characters (the raw Location column is literally wrapped in single
    quotes, e.g. "'Imadol, L'"), then upper-cases the result so that
    'n' and 'N' -- or a Location parsed with slightly different
    whitespace -- are always treated as the same category.
    """
    return text.strip().strip("'\"").strip().upper()


def parse_location(raw_location: str) -> tuple[str, str]:
    """
    Splits a raw Location string into (area, district_code).

    Example
    -------
    "'Imadol, L'"  ->  ("IMADOL", "L")

    The district fragment is reduced to letters only and the first
    letter is taken, so both a bare code ("L") and a stray trailing
    quote ("L'") resolve to the same district code. If no comma is
    present, or the district fragment doesn't match a known code, the
    district is reported as "UNKNOWN" so it still flows through the
    pipeline (falling back to the global mean price) instead of
    crashing the whole run over one malformed row.
    """
    text = normalize_category(raw_location)

    if "," in text:
        area_part, district_part = text.rsplit(",", 1)
    else:
        area_part, district_part = text, ""

    area = area_part.strip().strip("'\"").strip()

    letters_only = re.sub(r"[^A-Z]", "", district_part)
    district_code = letters_only[0] if letters_only else "UNKNOWN"
    if district_code not in DISTRICT_NAMES:
        district_code = "UNKNOWN"

    return area, district_code


def load_dataset_rows(file_path: str) -> tuple[list[dict[str, Any]], str]:
    """
    Reads one Project DARA CSV file and returns parsed row dictionaries
    plus the detected column schema.

    Parameters
    ----------
    file_path : str
        Path to any of the 8 Project DARA dataset CSV files.

    Returns
    -------
    rows : list[dict]
        One dict per data row with keys: 'area', 'district', 'land_area',
        'bedrooms', 'facing', 'road_access', 'price', and EITHER 'floor'
        (floor schema) OR 'interaction' (interaction schema).
    schema : str
        'floor' if the file has a Floor column (raw / outliers files),
        'interaction' if it has a Land Area x Road Access column
        (ghost / final files).

    Raises
    ------
    ValueError
        If the file's header matches neither known schema.
    """
    rows: list[dict[str, Any]] = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if "Floor" in fieldnames:
            schema = "floor"
        elif "Land Area x Road Access" in fieldnames:
            schema = "interaction"
        else:
            raise ValueError(
                f"Unrecognized column schema in {file_path}: {fieldnames}"
            )

        for raw_row in reader:
            if not raw_row or not raw_row.get("Location"):
                continue  # Skip blank trailing lines

            area, district = parse_location(raw_row["Location"])

            row: dict[str, Any] = {
                "area": area,
                "district": district,
                "land_area": float(raw_row["Land Area"]),
                "bedrooms": float(raw_row["Bedrooms"]),
                "facing": normalize_category(raw_row["Facing"]),
                "road_access": float(raw_row["Road Access"]),
                "price": float(raw_row["Price"]),
            }

            if schema == "floor":
                row["floor"] = float(raw_row["Floor"])
            else:
                row["interaction"] = float(raw_row["Land Area x Road Access"])

            rows.append(row)

    return rows, schema


def split_indices(n: int, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffles row indices 0..n-1 and splits them into train/test index
    arrays according to `train_ratio`.
    """
    indices = np.random.permutation(n)
    cutoff = int(train_ratio * n)
    return indices[:cutoff], indices[cutoff:]


def build_area_district_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    """
    Builds a lookup from Area -> the District it belongs to, using every
    row in the file (a given Area always belongs to one District, so this
    is safe to build from the full file rather than the training split).
    Used to give a rare Area a District-level fallback price estimate.
    """
    mapping: dict[str, str] = {}
    for row in rows:
        mapping.setdefault(row["area"], row["district"])
    return mapping


def build_price_encoding(
    rows: list[dict[str, Any]],
    key: str,
    train_idx: np.ndarray,
    global_mean_price: float,
    smoothing: float,
    prior_lookup: Optional[dict[str, float]] = None,
    prior_key_map: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """
    Computes a smoothed, price-based (target) encoding for a categorical
    column, fit on the TRAINING split only.

    encoded_value = (n * category_mean_price + smoothing * prior)
                     / (n + smoothing)

    Parameters
    ----------
    rows : list[dict]
        All parsed rows (row['price'] must be present).
    key : str
        Which categorical field to encode: 'district', 'area', or 'facing'.
    train_idx : np.ndarray
        Indices of rows belonging to the training split. Only these rows
        contribute to the encoding, to avoid leaking test-set prices into
        a feature that predicts price.
    global_mean_price : float
        Overall training-set mean price, used as the prior when no
        hierarchical `prior_lookup` is supplied (District, Facing).
    smoothing : float
        Larger values pull rare categories more strongly toward the
        prior; smaller values trust the category's own sample mean more.
    prior_lookup, prior_key_map : dict, optional
        Used for Area only: `prior_key_map` maps an Area -> its District,
        and `prior_lookup` maps a District -> its own price code, so a
        rarely-seen Area is anchored to its District's price level
        instead of the flat global mean.

    Returns
    -------
    dict[str, float]
        Mapping from category value (as seen in the training split) to
        its smoothed price code.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for i in train_idx:
        category = rows[i][key]
        sums[category] = sums.get(category, 0.0) + rows[i]["price"]
        counts[category] = counts.get(category, 0) + 1

    encoding: dict[str, float] = {}
    for category, total in sums.items():
        n = counts[category]
        category_mean = total / n

        if prior_lookup is not None and prior_key_map is not None:
            parent = prior_key_map.get(category)
            prior = prior_lookup.get(parent, global_mean_price)
        else:
            prior = global_mean_price

        encoding[category] = (n * category_mean + smoothing * prior) / (n + smoothing)

    return encoding


def process_dataset(file_path: str, train_ratio: float) -> dict[str, Any]:
    """
    Runs the full preprocessing pipeline on a single Project DARA CSV
    file: load -> parse Location -> split -> fit price encodings on the
    training split -> encode every row -> min-max scale (fit on training
    split) -> return everything downstream code needs.

    Returns
    -------
    dict with keys:
        'X_train', 'X_test', 'y_train', 'y_test' : np.ndarray
        'feature_names' : list[str]  (column order of X)
        'schema' : 'floor' or 'interaction'
        'district_price_map', 'area_price_map', 'facing_price_map' : dict
        'area_district_map' : dict[str, str]
        'global_mean_price' : float
        'X_min', 'X_max' : np.ndarray  (training-split scaling bounds)
        'districts_available' : list[str]
        'areas_by_district' : dict[str, list[str]]
        'facings_available' : list[str]
    """
    # Reset the seed per file so each dataset's shuffle/split is
    # reproducible independently of how many files were processed before it.
    np.random.seed(42)

    rows, schema = load_dataset_rows(file_path)
    n = len(rows)
    train_idx, test_idx = split_indices(n, train_ratio)

    global_mean_price = float(np.mean([rows[i]["price"] for i in train_idx]))

    district_price_map = build_price_encoding(
        rows, "district", train_idx, global_mean_price, SMOOTHING_DISTRICT
    )

    area_district_map = build_area_district_map(rows)
    area_price_map = build_price_encoding(
        rows, "area", train_idx, global_mean_price, SMOOTHING_AREA,
        prior_lookup=district_price_map, prior_key_map=area_district_map,
    )

    facing_price_map = build_price_encoding(
        rows, "facing", train_idx, global_mean_price, SMOOTHING_FACING
    )

    def encode_row(row: dict[str, Any]) -> list[float]:
        district_code = district_price_map.get(row["district"], global_mean_price)

        if row["area"] in area_price_map:
            area_code = area_price_map[row["area"]]
        else:
            # Unseen Area (can happen for a row that landed in the test
            # split): fall back to its District's price level.
            area_code = district_price_map.get(row["district"], global_mean_price)

        facing_code = facing_price_map.get(row["facing"], global_mean_price)
        extra = row["floor"] if schema == "floor" else row["interaction"]

        return [
            district_code,
            area_code,
            row["land_area"],
            row["bedrooms"],
            row["road_access"],
            extra,
            facing_code,
        ]

    feature_names = [
        "District_PriceCode",
        "Area_PriceCode",
        "Land Area",
        "Bedrooms",
        "Road Access",
        "Floor" if schema == "floor" else "Land Area x Road Access",
        "Facing_PriceCode",
    ]

    X_all = np.array([encode_row(r) for r in rows], dtype=np.float64)
    y_all = np.array([r["price"] for r in rows], dtype=np.float64)

    X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Fit scaling bounds on the TRAINING split only, apply to both splits.
    X_min = X_train_raw.min(axis=0)
    X_max = X_train_raw.max(axis=0)
    range_diff = X_max - X_min
    range_diff[range_diff == 0] = 1.0

    X_train = (X_train_raw - X_min) / range_diff
    X_test = (X_test_raw - X_min) / range_diff

    # Menu-building helpers for the live prediction step: only Areas that
    # actually received a training-set price code are offered as valid
    # choices, grouped under the District they belong to.
    areas_by_district: dict[str, list[str]] = {}
    for area in area_price_map:
        district = area_district_map[area]
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
        "district_price_map": district_price_map,
        "area_price_map": area_price_map,
        "facing_price_map": facing_price_map,
        "area_district_map": area_district_map,
        "global_mean_price": global_mean_price,
        "X_min": X_min,
        "X_max": X_max,
        "districts_available": sorted(district_price_map.keys()),
        "areas_by_district": areas_by_district,
        "facings_available": sorted(facing_price_map.keys()),
    }


def run_preprocessing_pipeline(base_path: str = ".") -> dict[str, dict[str, Any]]:
    """
    Runs the full Project DARA preprocessing pipeline on all EIGHT
    dataset files and returns a dictionary keyed by dataset label:

        'raw_1', 'outliers_1', 'ghost_1', 'final_1',
        'raw_2', 'outliers_2', 'ghost_2', 'final_2'

    Each value is the dict returned by `process_dataset` for that file.

    Parameters
    ----------
    base_path : str, default "."
        Directory containing the 8 CSV files.

    Returns
    -------
    dict[str, dict[str, Any]]
    """
    results: dict[str, dict[str, Any]] = {}
    for label, (filename, train_ratio) in DATASET_FILES.items():
        path = filename if base_path in (".", "") else f"{base_path}/{filename}"
        results[label] = process_dataset(path, train_ratio)
    return results


# ==========================================
# MAIN PREPROCESSING PIPELINE ENGINE EXECUTION
# ==========================================
if __name__ == "__main__":
    # Demonstrates the pipeline running standalone. Runs only when this
    # file is executed directly, never on `import Preprocessing`.
    print("--- Running Project DARA Preprocessing Pipeline on all 8 datasets ---\n")
    all_data = run_preprocessing_pipeline()

    for label, ds in all_data.items():
        print(f"[{label}]  schema={ds['schema']}  "
              f"X_train={ds['X_train'].shape}  X_test={ds['X_test'].shape}")
        print(f"    Districts: {ds['districts_available']}")
        print(f"    Unique Areas encoded: {len(ds['area_price_map'])}")
        print(f"    Facings: {ds['facings_available']}")
        print(f"    Global mean price (train): {ds['global_mean_price']:,.2f}\n")

    print("Processing finished. Every dataset's matrices are ready for the regression engines.")
