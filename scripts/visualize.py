import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Shared constants ──────────────────────────────────────────────────
PROCESSED_DATA = 'earthquake_project/data/processed/cleaned_earthquakes.csv'
FIGURES_DIR = 'earthquake_project/results/figures'

# Severity order for mag_category bar chart
SEVERITY_ORDER = [
    'Minor (2.5-3)',
    'Light (3-4)',
    'Moderate (4-5)',
    'Strong (5-6)',
    'Major (6-7)',
    'Great (7-8)',
    'Massive (8+)',
]


def run_visualizations():
    print("=" * 60)
    print("  Data Visualizations")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DATA)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Class Distribution ─────────────────────────────────────
    print("  Creating class distribution chart ...")
    counts = df['mag_category'].value_counts()

    # Reindex by severity order (keep only categories present in data)
    ordered_cats = [c for c in SEVERITY_ORDER if c in counts.index]
    # Append any categories not in our predefined order
    for c in counts.index:
        if c not in ordered_cats:
            ordered_cats.append(c)
    counts = counts.reindex(ordered_cats)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=len(counts) - 1)
    colors = [cmap(norm(i)) for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white',
                  linewidth=0.8)

    # Add count labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f'{int(val):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Magnitude Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Earthquake Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── 2. Feature Correlation Heatmap ────────────────────────────
    print("  Creating feature correlation heatmap ...")
    exclude_corr = ['mag', 'mag_category_encoded']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_corr]

    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            text_color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=6)

    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_correlation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── 3. Geographic Distribution ────────────────────────────────
    print("  Creating geographic distribution scatter ...")
    sample_size = min(50_000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    fig, ax = plt.subplots(figsize=(14, 7))
    scatter = ax.scatter(
        df_sample['longitude'], df_sample['latitude'],
        c=df_sample['mag_category_encoded'],
        cmap='YlOrRd', s=0.5, alpha=0.6,
    )
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Geographic Distribution of Earthquakes (colored by magnitude category)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(alpha=0.2)
    plt.colorbar(scatter, ax=ax, label='Magnitude Category (encoded)', shrink=0.8)
    plt.savefig(os.path.join(FIGURES_DIR, 'geographic_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── 4. Earthquakes per Year ───────────────────────────────────
    print("  Creating earthquakes per year line plot ...")
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
    else:
        # Fallback if year column not available
        year_counts = pd.Series(dtype=int)

    if not year_counts.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(year_counts.index, year_counts.values,
                color='#1976D2', linewidth=1.5)
        ax.fill_between(year_counts.index, year_counts.values,
                        alpha=0.15, color='#1976D2')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Earthquakes', fontsize=12)
        ax.set_title('Earthquakes per Year', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(FIGURES_DIR, 'earthquakes_per_year.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("    ⚠ 'year' column not found — skipping earthquakes_per_year plot.")

    print(f"\n  All visualizations saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    run_visualizations()
