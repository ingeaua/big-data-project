"""
Dimension Reduction Script for Earthquake Classification Project.

Performs PCA, ICA, and t-SNE on the cleaned earthquake data.
Saves transformed datasets, fitted model objects, and visualizations.
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — MUST be before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE

# ── Shared constants ──────────────────────────────────────────────
PROCESSED_DATA = 'earthquake_project/data/processed/cleaned_earthquakes.csv'
EXCLUDE_COLS = ['mag', 'mag_category', 'mag_category_encoded']
TEST_SIZE = 0.2
RANDOM_STATE = 42
FIGURES_DIR = 'earthquake_project/results/figures'
DIM_RED_DIR = 'earthquake_project/results/dimension_reduction'


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DIM_RED_DIR, exist_ok=True)

    # ── 1. Load processed data ────────────────────────────────────
    print("Loading processed data ...")
    df = pd.read_csv(PROCESSED_DATA)
    print(f"  Shape: {df.shape}")

    # ── 2. Extract features & target ──────────────────────────────
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df['mag_category_encoded'].values

    # Keep mag_category strings for plot legends
    mag_category_strings = df['mag_category'].values

    # Drop constant columns (std == 0)
    non_const = X.columns[X.std() > 0]
    dropped = [c for c in X.columns if c not in non_const]
    if dropped:
        print(f"  Dropped constant columns: {dropped}")
    X = X[non_const]
    print(f"  Feature columns ({len(X.columns)}): {list(X.columns)}")

    # ── 3. Train / test split (SAME as all model scripts) ────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Also split mag_category strings for legend labels
    cat_train, cat_test = train_test_split(
        mag_category_strings, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 4. Scale ──────────────────────────────────────────────────
    print("Scaling features ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)
    joblib.dump(scaler, os.path.join(DIM_RED_DIR, 'scaler.joblib'))
    print("  Saved scaler.joblib")

    # ── 5. PCA ────────────────────────────────────────────────────
    print("Running PCA (n_components=2) ...")
    pca = PCA(n_components=2)
    pca_train = pca.fit_transform(X_train_scaled)
    pca_test = pca.transform(X_test_scaled)
    pca_full = pca.transform(X_full_scaled)

    joblib.dump(pca, os.path.join(DIM_RED_DIR, 'pca_model.joblib'))
    print("  Saved pca_model.joblib")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")

    pd.DataFrame({
        'PCA1': pca_train[:, 0], 'PCA2': pca_train[:, 1], 'target': y_train
    }).to_csv(os.path.join(DIM_RED_DIR, 'pca_train.csv'), index=False)

    pd.DataFrame({
        'PCA1': pca_test[:, 0], 'PCA2': pca_test[:, 1], 'target': y_test
    }).to_csv(os.path.join(DIM_RED_DIR, 'pca_test.csv'), index=False)

    pd.DataFrame({
        'PCA1': pca_full[:, 0], 'PCA2': pca_full[:, 1]
    }).to_csv(os.path.join(DIM_RED_DIR, 'pca_results.csv'), index=False)
    print("  Saved pca_train.csv, pca_test.csv, pca_results.csv")

    # ── 6. ICA ────────────────────────────────────────────────────
    print("Running ICA (n_components=2) ...")
    ica = FastICA(n_components=2, random_state=RANDOM_STATE)
    ica_train = ica.fit_transform(X_train_scaled)
    ica_test = ica.transform(X_test_scaled)
    ica_full = ica.transform(X_full_scaled)

    joblib.dump(ica, os.path.join(DIM_RED_DIR, 'ica_model.joblib'))
    print("  Saved ica_model.joblib")

    pd.DataFrame({
        'ICA1': ica_train[:, 0], 'ICA2': ica_train[:, 1], 'target': y_train
    }).to_csv(os.path.join(DIM_RED_DIR, 'ica_train.csv'), index=False)

    pd.DataFrame({
        'ICA1': ica_test[:, 0], 'ICA2': ica_test[:, 1], 'target': y_test
    }).to_csv(os.path.join(DIM_RED_DIR, 'ica_test.csv'), index=False)

    pd.DataFrame({
        'ICA1': ica_full[:, 0], 'ICA2': ica_full[:, 1]
    }).to_csv(os.path.join(DIM_RED_DIR, 'ica_results.csv'), index=False)
    print("  Saved ica_train.csv, ica_test.csv, ica_results.csv")

    # ── 7. t-SNE (on a sample from training set) ─────────────────
    print("Running t-SNE (n_components=2) on 10k sample ...")
    n_sample = min(10_000, len(X_train_scaled))
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_train_scaled), size=n_sample, replace=False)
    X_tsne_input = X_train_scaled[sample_idx]
    y_tsne_sample = y_train[sample_idx]
    cat_tsne_sample = cat_train[sample_idx]

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    tsne_result = tsne.fit_transform(X_tsne_input)

    pd.DataFrame({
        'TSNE1': tsne_result[:, 0], 'TSNE2': tsne_result[:, 1]
    }).to_csv(os.path.join(DIM_RED_DIR, 'tsne_results_sample.csv'), index=False)
    print("  Saved tsne_results_sample.csv")

    # ── 8. Visualizations ─────────────────────────────────────────
    print("Creating visualizations ...")

    # Helper: build a consistent colormap for mag_category labels
    unique_labels = sorted(set(mag_category_strings))
    cmap = plt.cm.get_cmap('RdYlGn_r', len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    def _scatter_plot(coords, labels, title, xlabel, ylabel, filepath,
                      max_points=20_000):
        """2-D scatter coloured by mag_category string labels."""
        fig, ax = plt.subplots(figsize=(10, 8))
        if len(coords) > max_points:
            idx = rng.choice(len(coords), size=max_points, replace=False)
            coords = coords[idx]
            labels = labels[idx]

        for lab in unique_labels:
            mask = labels == lab
            if mask.any():
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[label_to_color[lab]], label=lab,
                           s=8, alpha=0.5, edgecolors='none')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title='Magnitude Category', fontsize=9,
                  title_fontsize=10, markerscale=3)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filepath}")

    # 8a. PCA explained variance bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    components = ['PC1', 'PC2']
    variances = pca.explained_variance_ratio_
    bars = ax.bar(components, variances, color=['#2196F3', '#FF9800'],
                  edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold')
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('PCA Explained Variance per Component', fontsize=14,
                 fontweight='bold')
    ax.set_ylim(0, max(variances) * 1.2)
    plt.savefig(os.path.join(FIGURES_DIR, 'pca_explained_variance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURES_DIR}/pca_explained_variance.png")

    # 8b. PCA scatter
    _scatter_plot(pca_train, cat_train,
                  'PCA — 2-D Projection (train, 20k sample)',
                  'PCA1', 'PCA2',
                  os.path.join(FIGURES_DIR, 'pca_scatter.png'))

    # 8c. ICA scatter
    _scatter_plot(ica_train, cat_train,
                  'ICA — 2-D Projection (train, 20k sample)',
                  'ICA1', 'ICA2',
                  os.path.join(FIGURES_DIR, 'ica_scatter.png'))

    # 8d. t-SNE scatter
    _scatter_plot(tsne_result, cat_tsne_sample,
                  't-SNE — 2-D Projection (10k sample)',
                  'TSNE1', 'TSNE2',
                  os.path.join(FIGURES_DIR, 'tsne_scatter.png'),
                  max_points=n_sample)

    print("\nDimension reduction complete.")


if __name__ == '__main__':
    main()
