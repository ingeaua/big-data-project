"""
Data Preparation Script for Earthquake Classification Project.

Loads raw earthquake data, removes leaky/post-hoc features,
handles missing values, encodes categoricals, and saves
the cleaned dataset for downstream model training.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ── Shared constants ──────────────────────────────────────────────
RAW_DATA = 'earthquake_project/data/raw/earthquakes_1900_2026.csv'
PROCESSED_DATA = 'earthquake_project/data/processed/cleaned_earthquakes.csv'

# Columns determined AFTER the earthquake measurement (leaky / identifiers)
LEAKY_COLS = [
    'id', 'time', 'place', 'updated', 'net',   # identifiers / text
    'magType',                                    # measurement method, depends on mag
    'dmin',                                      # measurement quality metrics (post-hoc)
    'status',                                     # reviewed/automatic (post-hoc)
    'depth_category',                             # derived directly from depth (redundant)
]


def main():
    # ── 1. Load raw data ──────────────────────────────────────────
    print("Loading raw data ...")
    df = pd.read_csv(RAW_DATA)
    print(f"  Raw shape: {df.shape}")

    # ── 1.5 Extract time features ─────────────────────────────────
    print("  Extracting additional time features ...")
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second
    df['day_of_week'] = df['time'].dt.dayofweek

    # ── 2. Drop leaky columns ─────────────────────────────────────
    cols_to_drop = [c for c in LEAKY_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped leaky columns: {cols_to_drop}")
    print(f"  Shape after drop: {df.shape}")

    # ── 3. Drop rows where mag_category is missing ────────────────
    before = len(df)
    df.dropna(subset=['mag_category'], inplace=True)
    print(f"  Dropped {before - len(df)} rows with missing mag_category")

    # ── 4. Handle missing values ──────────────────────────────────
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    # mag_category is categorical but will be encoded separately
    categorical_cols = [c for c in categorical_cols if c != 'mag_category']

    for col in numeric_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"    Filled {n_miss} missing values in '{col}' with median")

    for col in categorical_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            df[col] = df[col].fillna('unknown')
            print(f"    Filled {n_miss} missing values in '{col}' with 'unknown'")

    # ── 5. Encode categorical features ────────────────────────────
    # After removing leaky columns, only 'type' should remain as categorical
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"  Encoded '{col}': {list(le.classes_)}")

    # ── 6. Encode target (mag_category) ───────────────────────────
    target_le = LabelEncoder()
    df['mag_category_encoded'] = target_le.fit_transform(df['mag_category'])
    print("\n  Target mapping (mag_category -> encoded):")
    for i, cls in enumerate(target_le.classes_):
        print(f"    {cls} -> {i}")

    # ── 7. Save processed data ────────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"\n  Saved processed data to: {PROCESSED_DATA}")
    print(f"  Final shape: {df.shape}")

    # ── 8. Print final feature list ───────────────────────────────
    exclude = ['mag', 'mag_category', 'mag_category_encoded']
    features = [c for c in df.columns if c not in exclude]
    print(f"\n  Final features ({len(features)}): {features}")
    print("  Expected: year, month, day_of_year, hour, latitude, longitude, depth, type, tsunami")


if __name__ == '__main__':
    main()
