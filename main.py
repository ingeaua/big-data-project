"""
Earthquake Project — Full Pipeline Orchestrator
=================================================
Runs all project phases in sequence:
  1. Data Preparation
  2. Data Visualization
  3. Dimension Reduction
  4. Model Training (5 models)
  5. Model Evaluation
  6. Dimension Reduction Comparison
"""

import subprocess
import time
import sys


def _run_step(description, script_path):
    """Run a single pipeline step, abort on failure."""
    print(f"\n  Running: {script_path}")
    subprocess.run([sys.executable, script_path], check=True)
    print(f"  ✓ {description} complete.")


def run_pipeline():
    start = time.time()

    print("=" * 64)
    print("  EARTHQUAKE CLASSIFICATION PIPELINE")
    print("=" * 64)

    # ── Phase 1: Data Preparation ─────────────────────────────────
    print("\n" + "─" * 64)
    print("  Phase 1: Data Preparation")
    print("─" * 64)
    _run_step("Data Preparation",
              "earthquake_project/scripts/data_prep.py")

    # ── Phase 2: Data Visualization ───────────────────────────────
    print("\n" + "─" * 64)
    print("  Phase 2: Data Visualization")
    print("─" * 64)
    _run_step("Data Visualization",
              "earthquake_project/scripts/visualize.py")

    # ── Phase 3: Dimension Reduction ──────────────────────────────
    print("\n" + "─" * 64)
    print("  Phase 3: Dimension Reduction")
    print("─" * 64)
    _run_step("Dimension Reduction",
              "earthquake_project/scripts/dimension_reduction.py")

    # ── Phase 4: Model Training ───────────────────────────────────
    print("\n" + "─" * 64)
    print("  Phase 4: Model Training (5 models)")
    print("─" * 64)
    model_scripts = [
        "logistic_regression.py",
        "random_forest.py",
        "gradient_boosting.py",
        "svm.py",
        "knn.py",
    ]
    for script in model_scripts:
        _run_step(script.replace('.py', ''),
                  f"earthquake_project/scripts/models/{script}")

    # ── Phase 5: Model Evaluation ─────────────────────────────────
    print("\n" + "─" * 64)
    print("  Phase 5: Model Evaluation")
    print("─" * 64)
    _run_step("Model Evaluation",
              "earthquake_project/scripts/evaluate_results.py")

    # ── Phase 6: Dimension Reduction Comparison ───────────────────
    print("\n" + "─" * 64)
    print("  Phase 6: Dimension Reduction Comparison")
    print("─" * 64)
    _run_step("Dimension Reduction Comparison",
              "earthquake_project/scripts/dim_reduction_comparison.py")

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    minutes, seconds = divmod(elapsed, 60)
    print("\n" + "=" * 64)
    print("  Pipeline completed successfully!")
    print(f"  Total elapsed time: {int(minutes)}m {seconds:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    run_pipeline()
