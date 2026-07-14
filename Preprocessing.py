"""
preprocessing.py
-----------------
Preprocessing module for Project DARA (Design and Analysis of Regression
Algorithms).

This module is the single gateway between the raw, already-cleaned
Kathmandu Valley housing CSV files and every regression algorithm in the
project (Simple Linear, Multiple Linear, Polynomial, Ridge). It is
responsible for:

    1. Loading the cleaned CSV datasets.
    2. Discovering unique categorical values (Location, Facing).
    3. Building encoding dictionaries (and their reverse counterparts).
    4. Converting categorical variables into numerical codes.
    5. Converting the dataset into NumPy matrices.
    6. Separating features (X) from the target (y = Price).
    7. Normalizing feature values (min-max scaling).
    8. Shuffling the dataset.
    9. Splitting into training and testing sets.
    10. Returning processed matrices ready for the regression engines.

No missing-value handling, duplicate removal, or unit conversion is
performed here -- that cleaning has already been done manually on the
source CSV files, per the Project DARA proposal. The only "cleaning-like"
step in this module is categorical string normalization (see
`normalize_category`), which exists purely to make sure the *encoding*
step treats equivalent category labels (e.g. 'N' and 'n') as the same
category -- it does not drop, alter, or invent any data.

Design note for the rest of the team:
    Importing this module does NOT execute the pipeline. Call
    `run_preprocessing_pipeline(...)` explicitly (e.g. from main.py) to
    get back a dictionary of ready-to-use NumPy matrices.
"""

import csv
from typing import Any

import numpy as np

# Fixed seed so every teammate gets the same shuffle/split when they run
# the pipeline -- required for reproducible comparisons across the four
# regression models.
np.random.seed(42)


def normalize_category(text: str) -> str:
    """
    Normalizes a raw categorical string for consistent encoding.

    Parameters
    ----------
    text : str
        Raw string value from a categorical column (Location or Facing).

    Returns
    -------
    str
        The string with surrounding whitespace removed and converted to
        uppercase, e.g. ' n ' -> 'N'.

    Why this exists
    ----------------
    The dataset already went through manual cleaning (missing values,
    duplicates, unit standardization), but casing was not part of that
    pass. Without this step, 'N' and 'n' would be assigned two different
    numeric codes by `encode_to_matrix`, which is a categorical-encoding
    bug, not a data-cleaning issue -- the underlying category is the
    same, only its text representation differs. Applying this uppercase
    normalization consistently, both when discovering categories and
    when encoding rows, guarantees a 1-to-1 mapping between real-world
    categories and numeric codes.
    """
    return text.strip().upper()


def load_and_gather_categories(
    file_path: str,
) -> tuple[list[list[str]], set[str], set[str]]:
    """
    Reads a CSV file using vanilla structures and returns rows along with
    sets of unique, normalized strings discovered in the text columns.

    Parameters
    ----------
    file_path : str
        Path to a cleaned housing dataset CSV file. Expected column
        layout: 0=Location, 1=Land Area, 2=Bedrooms, 3=Floor, 4=Facing,
        5=Road Access, 6=Price.

    Returns
    -------
    rows : list[list[str]]
        Raw row data exactly as read from the CSV (unnormalized), kept
        for later use by `encode_to_matrix`.
    locations : set[str]
        Unique, normalized (stripped + uppercased) Location strings found
        in this file.
    facings : set[str]
        Unique, normalized (stripped + uppercased) Facing strings found
        in this file.
    """
    rows: list[list[str]] = []
    locations: set[str] = set()
    facings: set[str] = set()

    with open(file_path, mode='r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Automatically skip the header row

        for row in reader:
            if not row:
                continue  # Skip any trailing blank lines

            # Column layout matching your files:
            # 0: Location, 1: Land Area, 2: Bedrooms, 3: Floor, 4: Facing, 5: Road Access, 6: Price
            locations.add(normalize_category(row[0]))
            facings.add(normalize_category(row[4]))
            rows.append(row)

    return rows, locations, facings


def encode_to_matrix(
    raw_rows: list[list[str]],
    loc_map: dict[str, int],
    fac_map: dict[str, int],
) -> np.ndarray:
    """
    Maps text features using encoding dictionaries and extracts numerical
    values into a structured numpy matrix layout.

    Parameters
    ----------
    raw_rows : list[list[str]]
        Raw row data as returned by `load_and_gather_categories`.
    loc_map : dict[str, int]
        Mapping from normalized Location string to its numeric code.
    fac_map : dict[str, int]
        Mapping from normalized Facing string to its numeric code.

    Returns
    -------
    np.ndarray, shape (n_rows, 7), dtype float64
        Columns, in order: [location_code, land_area, bedrooms, floor,
        facing_code, road_access, price]. Price remains the last column
        so callers can split X/y themselves.

    Note
    ----
    `loc_map` / `fac_map` must be built on normalized strings (see
    `normalize_category`). The lookup here normalizes each raw value the
    same way before indexing, so a casing difference between how a
    category was discovered and how it appears in a given row never
    causes a KeyError or a silent mismatch.
    """
    processed_data: list[list[float]] = []
    for row in raw_rows:
        # Convert text columns using maps, convert standard columns directly to floats
        loc_code = float(loc_map[normalize_category(row[0])])
        land_area = float(row[1])
        bedrooms = float(row[2])
        floor = float(row[3])
        facing_code = float(fac_map[normalize_category(row[4])])
        road_access = float(row[5])
        price = float(row[6])

        processed_data.append([loc_code, land_area, bedrooms, floor, facing_code, road_access, price])

    return np.array(processed_data, dtype=np.float64)


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    """
    Inverts a forward label -> code dictionary into a code -> label dictionary.

    Parameters
    ----------
    mapping : dict[str, int]
        A forward encoding dictionary, e.g. `location_map`, mapping a
        normalized category string to its integer code.

    Returns
    -------
    dict[int, str]
        The reverse mapping (code -> label), so any module downstream
        (e.g. main.py, when printing predictions or plot labels) can
        convert an encoded Location or Facing value back into readable
        text.
    """
    return {code: label for label, code in mapping.items()}


def min_max_scale_features(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs feature scaling from scratch to bind values perfectly between 0 and 1.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (Location, Land Area, Bedrooms, Floor, Facing,
        Road Access) before scaling. Must NOT include the Price column.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Feature matrix with every column linearly rescaled to [0, 1].
    X_min : np.ndarray, shape (n_features,)
        Per-column minimum used for scaling (kept so a new/unseen sample
        could be scaled identically later, e.g. for a single prediction).
    X_max : np.ndarray, shape (n_features,)
        Per-column maximum used for scaling.
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    # Handle edge case where max equals min to prevent division by zero
    range_diff = X_max - X_min
    range_diff[range_diff == 0] = 1.0

    X_scaled = (X - X_min) / range_diff
    return X_scaled, X_min, X_max


def shuffle_and_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffles data indices randomly and splits them based on a custom train/test ratio.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target values (Price, in NPR, left unscaled).
    train_ratio : float
        Fraction of samples assigned to the training set (e.g. 0.70 for
        a 70/30 split, 0.80 for an 80/20 split).

    Returns
    -------
    X_train, X_test : np.ndarray
        Feature matrices for training and testing.
    y_train, y_test : np.ndarray
        Target vectors for training and testing, aligned with X_train /
        X_test respectively.
    """
    num_samples = X.shape[0]
    shuffled_indices = np.random.permutation(num_samples)

    # Calculate cutoff boundary integer index
    split_boundary = int(train_ratio * num_samples)

    train_idx = shuffled_indices[:split_boundary]
    test_idx = shuffled_indices[split_boundary:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(
    dataset_1_path: str,
    dataset_2_path: str,
    train_ratio_1: float = 0.70,
    train_ratio_2: float = 0.80,
) -> dict[str, Any]:
    """
    Runs the full Project DARA preprocessing pipeline end-to-end and
    returns everything the regression engines (and main.py) need.

    This is the single entry point the rest of the team should import
    and call. Per the Project DARA architecture, regression engines and
    main.py should never read a CSV file directly -- raw data flows
    through this module only.

    Parameters
    ----------
    dataset_1_path : str
        Path to the cleaned Dataset 1 CSV file.
    dataset_2_path : str
        Path to the cleaned Dataset 2 CSV file.
    train_ratio_1 : float, default 0.70
        Train/test split ratio for Dataset 1 (proposal: 70/30).
    train_ratio_2 : float, default 0.80
        Train/test split ratio for Dataset 2 (proposal: 80/20).

    Returns
    -------
    dict[str, Any]
        {
            'X1_train', 'X1_test', 'y1_train', 'y1_test' : Dataset 1 matrices,
            'X2_train', 'X2_test', 'y2_train', 'y2_test' : Dataset 2 matrices,
            'location_map'          : dict[str, int]  normalized location -> code,
            'facing_map'            : dict[str, int]  normalized facing  -> code,
            'reverse_location_map'  : dict[int, str]  code -> location,
            'reverse_facing_map'    : dict[int, str]  code -> facing,
            'X1_min', 'X1_max'      : per-column scaling bounds for Dataset 1,
            'X2_min', 'X2_max'      : per-column scaling bounds for Dataset 2,
        }

    Notes
    -----
    Location and Facing vocabularies are unified across BOTH datasets
    before encoding, so a given location or facing always maps to the
    same numeric code regardless of which dataset it appears in. This
    matters if the team ever trains on one dataset and evaluates on
    the other.
    """
    # --- Step 1 & 2: Load files & extract text dictionaries ---
    rows_d1, locs_d1, facs_d1 = load_and_gather_categories(dataset_1_path)
    rows_d2, locs_d2, facs_d2 = load_and_gather_categories(dataset_2_path)

    # Build unified unique feature lists across both data collections
    all_unique_locations = sorted(list(locs_d1.union(locs_d2)))
    all_unique_facings = sorted(list(facs_d1.union(facs_d2)))

    # Create explicit 'Text-to-Number' dictionaries, and their reverse
    location_map = {loc: index for index, loc in enumerate(all_unique_locations)}
    facing_map = {facing: index for index, facing in enumerate(all_unique_facings)}
    reverse_location_map = invert_mapping(location_map)
    reverse_facing_map = invert_mapping(facing_map)

    # --- Transform strings and build numerical arrays ---
    matrix_d1 = encode_to_matrix(rows_d1, location_map, facing_map)
    matrix_d2 = encode_to_matrix(rows_d2, location_map, facing_map)

    # Isolate features (X) from final column target values (y - Price)
    X1, y1 = matrix_d1[:, :-1], matrix_d1[:, -1]
    X2, y2 = matrix_d2[:, :-1], matrix_d2[:, -1]

    # --- Step 3: NumPy conversion & min-max scaling from scratch ---
    X1_scaled, X1_min, X1_max = min_max_scale_features(X1)
    X2_scaled, X2_min, X2_max = min_max_scale_features(X2)

    # --- Step 4: Random shuffling & custom matrix splitting ---
    X1_train, X1_test, y1_train, y1_test = shuffle_and_split(X1_scaled, y1, train_ratio=train_ratio_1)
    X2_train, X2_test, y2_train, y2_test = shuffle_and_split(X2_scaled, y2, train_ratio=train_ratio_2)

    return {
        'X1_train': X1_train,
        'X1_test': X1_test,
        'y1_train': y1_train,
        'y1_test': y1_test,
        'X2_train': X2_train,
        'X2_test': X2_test,
        'y2_train': y2_train,
        'y2_test': y2_test,
        'location_map': location_map,
        'facing_map': facing_map,
        'reverse_location_map': reverse_location_map,
        'reverse_facing_map': reverse_facing_map,
        'X1_min': X1_min,
        'X1_max': X1_max,
        'X2_min': X2_min,
        'X2_max': X2_max,
    }


# ==========================================
# MAIN PREPROCESSING PIPELINE ENGINE EXECUTION
# ==========================================
if __name__ == "__main__":
    # This block only demonstrates how to call the pipeline. It runs
    # when this file is executed directly (`python preprocessing.py`),
    # but never when another module does `import preprocessing` --
    # that was the core problem with the previous version.

    dataset_1_path = "Datasets_for_DARA_1.csv"
    dataset_2_path = "Datasets_for_DARA_2.csv"

    print("--- Step 1 & 2: Loading Files & Extracting Text Dictionaries ---")
    result = run_preprocessing_pipeline(dataset_1_path, dataset_2_path,
                                         train_ratio_1=0.70, train_ratio_2=0.80)

    print(f"Total Unique Locations Mapped: {len(result['location_map'])}")
    print(f"Total Unique House Facings Mapped: {len(result['facing_map'])}\n")

    print("--- Step 3: NumPy Conversion & Min-Max Scaling From Scratch ---")
    print("Features scaled successfully. Matrix limits verified.")
    print(
        f"Dataset 1 Features Matrix Min Bound check: {result['X1_train'].min()}, "
        f"Max Bound check: {result['X1_train'].max()}"
    )
    print(
        f"Dataset 2 Features Matrix Min Bound check: {result['X2_train'].min()}, "
        f"Max Bound check: {result['X2_train'].max()}\n"
    )

    print("--- Step 4: Random Shuffling & Custom Matrix Splitting ---")
    print("[DATASET 1 COMPLETE (70/30 split)]")
    print(f" -> X_train Shape: {result['X1_train'].shape} | y_train Shape: {result['y1_train'].shape}")
    print(f" -> X_test Shape:  {result['X1_test'].shape} | y_test Shape:  {result['y1_test'].shape}")

    print("\n[DATASET 2 COMPLETE (80/20 split)]")
    print(f" -> X_train Shape: {result['X2_train'].shape} | y_train Shape: {result['y2_train'].shape}")
    print(f" -> X_test Shape:  {result['X2_test'].shape} | y_test Shape:  {result['y2_test'].shape}")

    print("\nProcessing finished. Your pristine math matrices are ready for your regression models!")

