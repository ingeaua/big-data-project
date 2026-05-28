import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# ── Shared constants ──────────────────────────────────────────────────
PROCESSED_DATA = 'earthquake_project/data/processed/cleaned_earthquakes.csv'
EXCLUDE_COLS = ['mag', 'mag_category', 'mag_category_encoded']
TEST_SIZE = 0.2
RANDOM_STATE = 42
FIGURES_DIR = 'earthquake_project/results/figures'
DIM_RED_DIR = 'earthquake_project/results/dimension_reduction'
COMPARISON_CSV = 'earthquake_project/results/comparison_results.csv'
OUTPUT_CSV = 'earthquake_project/results/dim_reduction_comparison.csv'

# ── Model definitions ────────────────────────────────────────────────
MODEL_DEFS = {
    'Logistic Regression': lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    'Random Forest': lambda: RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

if XGBClassifier is not None:
    MODEL_DEFS['XGBoost'] = lambda: XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric='mlogloss',
    )

# Map model display names to keys that may appear in comparison_results.csv
ORIGINAL_NAME_MAP = {
    'Logistic Regression': ['logistic_regression', 'Logistic Regression'],
    'Random Forest': ['random_forest', 'Random Forest'],
    'XGBoost': ['gradient_boosting', 'XGBoost', 'xgboost', 'Gradient Boosting'],
}


def _load_dim_data(prefix):
    """Load train/test CSVs for a given reduction prefix (pca or ica).

    Tries <prefix>_train.csv / <prefix>_test.csv first.
    Falls back to <prefix>_results.csv + train_test_split.
    """
    train_path = os.path.join(DIM_RED_DIR, f'{prefix}_train.csv')
    test_path = os.path.join(DIM_RED_DIR, f'{prefix}_test.csv')

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  Loading {prefix.upper()} train/test from pre-split files ...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        feature_cols = [c for c in train.columns if c != 'target']
        return (
            train[feature_cols].values, test[feature_cols].values,
            train['target'].values, test['target'].values,
        )

    # Fallback: reconstruct from full results + original data
    print(f"  Pre-split files not found for {prefix.upper()}. "
          f"Reconstructing from {prefix}_results.csv ...")
    from sklearn.model_selection import train_test_split

    full_path = os.path.join(DIM_RED_DIR, f'{prefix}_results.csv')
    df_full = pd.read_csv(full_path)

    # Need target column — load from original data
    df_orig = pd.read_csv(PROCESSED_DATA)
    y_all = df_orig['mag_category_encoded'].values

    feature_cols = [c for c in df_full.columns if c != 'target']
    X_all = df_full[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def _train_and_eval(model_factory, X_train, X_test, y_train, y_test, scale=False):
    """Train model, optionally scale, return (accuracy, f1_weighted)."""
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return round(acc, 4), round(f1, 4)


def _get_original_scores(model_display_name):
    """Look up original accuracy & F1 from comparison_results.csv."""
    try:
        df = pd.read_csv(COMPARISON_CSV)
    except FileNotFoundError:
        return None, None

    candidates = ORIGINAL_NAME_MAP.get(model_display_name, [model_display_name])
    for name in candidates:
        row = df[df['Model'] == name]
        if not row.empty:
            acc = row.iloc[0].get('Accuracy', None)
            f1 = row.iloc[0].get('F1-Score (Weighted)', None)
            return acc, f1
    return None, None


def run_comparison():
    print("=" * 60)
    print("  Dimension Reduction — Model Comparison")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load reduced datasets ────────────────────────────────────
    pca_data = _load_dim_data('pca')
    ica_data = _load_dim_data('ica')

    rows = []
    for name, factory in MODEL_DEFS.items():
        print(f"\n  Model: {name}")

        # Determine if model needs scaling
        needs_scale = (name == 'Logistic Regression')

        pca_acc, pca_f1 = _train_and_eval(factory, *pca_data, scale=needs_scale)
        print(f"    PCA  → Acc={pca_acc:.4f}  F1={pca_f1:.4f}")

        ica_acc, ica_f1 = _train_and_eval(factory, *ica_data, scale=needs_scale)
        print(f"    ICA  → Acc={ica_acc:.4f}  F1={ica_f1:.4f}")

        orig_acc, orig_f1 = _get_original_scores(name)
        print(f"    Orig → Acc={orig_acc}  F1={orig_f1}")

        rows.append({
            'Model': name,
            'Original_Accuracy': orig_acc,
            'Original_F1': orig_f1,
            'PCA_Accuracy': pca_acc,
            'PCA_F1': pca_f1,
            'ICA_Accuracy': ica_acc,
            'ICA_F1': ica_f1,
        })

    comp_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("  Dimension Reduction Comparison")
    print("=" * 60)
    print(comp_df.to_string(index=False))

    comp_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")

    # ── Visualization ─────────────────────────────────────────────
    models = comp_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    orig_f1_vals = comp_df['Original_F1'].astype(float).values
    pca_f1_vals = comp_df['PCA_F1'].astype(float).values
    ica_f1_vals = comp_df['ICA_F1'].astype(float).values

    bars1 = ax.bar(x - width, orig_f1_vals, width, label='Original Features',
                   color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x, pca_f1_vals, width, label='PCA (2 components)',
                   color='#4CAF50', edgecolor='white')
    bars3 = ax.bar(x + width, ica_f1_vals, width, label='ICA (2 components)',
                   color='#FF9800', edgecolor='white')

    # Annotate bar values
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1-Score (Weighted)', fontsize=12)
    ax.set_title('F1-Score: Original vs PCA vs ICA Features',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(
        os.path.join(FIGURES_DIR, 'dim_reduction_comparison.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.close()
    print(f"Visualization saved to {FIGURES_DIR}/dim_reduction_comparison.png")


if __name__ == '__main__':
    run_comparison()
