import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

# ── Shared constants ──────────────────────────────────────────────────
PROCESSED_DATA = 'earthquake_project/data/processed/cleaned_earthquakes.csv'
EXCLUDE_COLS = ['mag', 'mag_category', 'mag_category_encoded']
TEST_SIZE = 0.2
RANDOM_STATE = 42
FIGURES_DIR = 'earthquake_project/results/figures'
MODELS_DIR = 'earthquake_project/results/models'


def _build_label_mapping(df):
    """Return ordered list of mag_category names sorted by encoded value."""
    pairs = df[['mag_category', 'mag_category_encoded']].drop_duplicates()
    pairs = pairs.sort_values('mag_category_encoded')
    return pairs['mag_category'].tolist()


def evaluate_models():
    print("=" * 60)
    print("  Model Evaluation — Full Test Set")
    print("=" * 60)

    # ── Load data & prepare features ──────────────────────────────
    df = pd.read_csv(PROCESSED_DATA)
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    y = df['mag_category_encoded']

    # Drop constant columns
    X = X.loc[:, X.std() > 0]

    # Same split as training scripts
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Category label mapping (sorted by encoded value)
    class_labels = _build_label_mapping(df)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Evaluate each model ───────────────────────────────────────
    model_files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib'))
    results = []
    per_class_f1 = {}

    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        print(f"\n  Evaluating {model_name} ...")

        loaded = joblib.load(os.path.join(MODELS_DIR, model_file))

        # Handle dict (model+scaler) vs bare model
        if isinstance(loaded, dict) and 'scaler' in loaded:
            model = loaded['model']
            scaler = loaded['scaler']
            X_proc = scaler.transform(X_test)
        else:
            model = loaded
            X_proc = X_test

        y_pred = model.predict(X_proc)

        # Overall metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        results.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Precision (Weighted)': round(prec, 4),
            'Recall (Weighted)': round(rec, 4),
            'F1-Score (Weighted)': round(f1, 4),
        })

        # Per-class F1 via classification_report
        report = classification_report(
            y_test, y_pred, output_dict=True
        )
        per_class = {}
        for encoded_val, label in enumerate(class_labels):
            key = str(encoded_val)
            if key in report:
                per_class[label] = round(report[key]['f1-score'], 4)
            else:
                per_class[label] = 0.0
        per_class_f1[model_name] = per_class

        # ── Confusion matrix figure ──────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=class_labels
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.savefig(
            os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name}.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close()

    # ── Results DataFrame ─────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values(
        by='F1-Score (Weighted)', ascending=False
    )
    print("\n" + "=" * 60)
    print("  Final Model Comparison (sorted by F1)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    output_csv = 'earthquake_project/results/comparison_results.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # ── Comparison bar chart ──────────────────────────────────────
    metrics = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
    short_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    models_sorted = results_df['Model'].tolist()
    x = np.arange(len(models_sorted))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, short_labels, colors)):
        vals = results_df.set_index('Model').loc[models_sorted, metric].values
        ax.bar(x + i * width, vals, width, label=label, color=color)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models_sorted, rotation=25, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(
        os.path.join(FIGURES_DIR, 'comparison_bar_chart.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.close()

    # ── Per-class F1 heatmap ──────────────────────────────────────
    heatmap_df = pd.DataFrame(per_class_f1).T  # rows = models, cols = classes
    # Reorder rows by overall F1 (best at top)
    heatmap_df = heatmap_df.loc[models_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels(heatmap_df.columns, rotation=30, ha='right')
    ax.set_yticklabels(heatmap_df.index)

    # Annotate cells
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.iloc[i, j]
            text_color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=10)

    ax.set_title('Per-Class F1-Score by Model', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.savefig(
        os.path.join(FIGURES_DIR, 'per_class_f1_heatmap.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.close()

    print(f"\nAll visualizations saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    evaluate_models()
