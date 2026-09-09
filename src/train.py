"""
train.py
========
Author : Ujjwal Deep
Project: Simple Demand Forecast (ML Basics - Day 1 to 3)

What this file does:
--------------------
This is the main script that ties everything together:

  Step 1 - Load the CSV data
  Step 2 - Add calendar + lag features (from features.py)
  Step 3 - Split data CHRONOLOGICALLY (80% train, 20% test)
           ** Never shuffle time-series data! Future data must not leak in **
  Step 4 - Scale features using StandardScaler
           ** IMPORTANT: fit the scaler on TRAIN data only **
  Step 5 - Train a LinearRegression model
  Step 6 - Evaluate on train and test sets (MAE, RMSE, R2)
  Step 7 - Generate and save 3 plots to the plots/ folder

Why LinearRegression?
---------------------
It's the simplest model that gives interpretable results.
We can look at the coefficients and say "feature X pushes demand up/down by Y".
For a first project, simple and understandable beats complex and opaque.

Usage:
------
  python src/generate_data.py   <- run first!
  python src/train.py
"""

import sys
import numpy as np
import pandas as pd

# Use 'Agg' backend so plots are saved to files (not shown on screen)
# This is needed when running as a script (not inside Jupyter)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── File paths ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent     # project root (one level up from src/)
DATA_PATH = ROOT / "data" / "demand_data.csv"
PLOT_DIR  = ROOT / "plots"

# Make sure src/ is importable (for features.py)
sys.path.insert(0, str(ROOT / "src"))

# ── Plot styling ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_prepare(path):
    """
    Load the CSV, then apply all feature engineering.
    Drop the first 14 rows that have NaN (from lag features).
    """
    from features import add_calendar_features, add_lag_features

    df = pd.read_csv(path, parse_dates=["date"])

    # Add calendar features (trend, day_of_week, month, etc.)
    df = add_calendar_features(df)

    # Add lag features (lag_7, lag_14, rolling_mean_7)
    df = add_lag_features(df)

    # Drop rows where lags couldn't be computed (first 14 rows)
    df = df.dropna().reset_index(drop=True)

    return df


# ==============================================================================
# TRAIN / TEST SPLIT
# ==============================================================================

def chronological_split(df, train_ratio=0.8):
    """
    Split data chronologically - first 80% for training, last 20% for testing.

    WHY NOT shuffle?
    In time-series, if we shuffle and put future rows in training,
    the model 'sees the future' during training - this is called LEAKAGE.
    It makes the model look great but it would fail on real future data.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    return train, test


# ==============================================================================
# METRICS
# ==============================================================================

def compute_metrics(y_true, y_pred, label=""):
    """
    Compute and print MAE, RMSE, R2.

    MAE  = Mean Absolute Error     → average error in units
    RMSE = Root Mean Squared Error → penalises large errors more than MAE
    R2   = how much variance the model explains (1.0 = perfect)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)

    tag = f"[{label}] " if label else ""
    print(f"  {tag}MAE  = {mae:.2f} units")
    print(f"  {tag}RMSE = {rmse:.2f} units")
    print(f"  {tag}R2   = {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_demand_overview(df):
    """Plot the full demand time series and its distribution."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: time series
    axes[0].plot(df["date"], df["demand"], lw=0.8, color="steelblue")
    axes[0].set_title("Daily Demand -- Full 2-Year History", fontweight="bold")
    axes[0].set_ylabel("Units Sold")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].tick_params(axis="x", rotation=30)

    # Bottom: distribution (histogram)
    axes[1].hist(df["demand"], bins=40, color="steelblue", edgecolor="white")
    axes[1].set_title("Demand Distribution", fontweight="bold")
    axes[1].set_xlabel("Units Sold")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    _save_plot("demand_overview.png")


def plot_actual_vs_predicted(train, test, y_train_pred, y_test_pred):
    """Plot actual demand vs model predictions for both train and test sets."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Training period
    ax.plot(train["date"], train["demand"], color="steelblue", lw=0.8,
            label="Train -- Actual")
    ax.plot(train["date"], y_train_pred, color="orange", lw=0.8, alpha=0.75,
            label="Train -- Predicted")

    # Test period
    ax.plot(test["date"], test["demand"], color="green", lw=1.2,
            label="Test -- Actual")
    ax.plot(test["date"], y_test_pred, color="red", lw=1.2, linestyle="--",
            label="Test -- Predicted")

    # Shade the test period to make it obvious
    ax.axvspan(test["date"].iloc[0], test["date"].iloc[-1],
               alpha=0.06, color="red", label="Test Period")

    ax.set_title("Actual vs Predicted Demand (Linear Regression)", fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save_plot("actual_vs_predicted.png")


def plot_residuals(test, y_test_pred):
    """
    Plot residuals (errors) over time and as a distribution.
    Residual = Actual - Predicted

    GOOD residuals: randomly scattered around 0, no pattern
    BAD residuals: show a pattern (model is missing something systematic)
    """
    residuals = test["demand"].values - y_test_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: residuals over time
    axes[0].plot(test["date"], residuals, lw=0.9, color="purple")
    axes[0].axhline(0, color="black", lw=1.2, linestyle="--")
    axes[0].set_title("Residuals Over Time -- Test Set", fontweight="bold")
    axes[0].set_ylabel("Actual - Predicted (units)")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].tick_params(axis="x", rotation=30)

    # Right: residual distribution
    axes[1].hist(residuals, bins=30, color="purple", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.2, linestyle="--")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual (units)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    _save_plot("residuals.png")


def plot_feature_importance(feature_names, coefficients):
    """
    Bar chart of Linear Regression coefficients.

    Since features are standardised before training, the coefficients
    are on the same scale — larger absolute value = more important.
    Blue bars push demand UP. Red bars push demand DOWN.
    """
    importance = pd.Series(coefficients, index=feature_names).sort_values()
    colors = ["crimson" if v < 0 else "steelblue" for v in importance]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance.index, importance.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Feature Coefficients (Standardised)", fontweight="bold")
    ax.set_xlabel("Coefficient Value")
    plt.tight_layout()
    _save_plot("feature_importance.png")


def _save_plot(filename):
    """Save current matplotlib figure to the plots/ directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    print("   [SAVED]", path)
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  SIMPLE DEMAND FORECAST -- Linear Regression")
    print("  by Ujjwal Deep | ML Basics Project")
    print("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        print("\n[ERROR] Data file not found:", DATA_PATH)
        print("  Please run: python src/generate_data.py\n")
        sys.exit(1)

    df = load_and_prepare(DATA_PATH)
    print(f"\n[1/7] Data loaded: {len(df)} rows x {len(df.columns)} columns")
    print("      (first 14 rows dropped due to lag NaN values)")

    # ── Step 2: Check for missing values ────────────────────────────────────
    missing = df.isnull().sum().sum()
    print(f"\n[2/7] Missing values: {missing}  (should be 0)")
    print("      Demand stats:")
    print("      ", df["demand"].describe().round(2).to_dict())

    # Generate overview plot
    plot_demand_overview(df)

    # ── Step 3: Feature / target setup ──────────────────────────────────────
    from features import get_feature_columns
    FEATURES = get_feature_columns()
    TARGET   = "demand"

    # ── Step 4: Chronological train/test split (80/20) ───────────────────────
    train, test = chronological_split(df, train_ratio=0.80)
    print(f"\n[3/7] Train/Test split (80/20 chronological):")
    print(f"      Train: {train['date'].min().date()} -> {train['date'].max().date()} ({len(train)} rows)")
    print(f"      Test : {test['date'].min().date()} -> {test['date'].max().date()} ({len(test)} rows)")

    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_test  = test[FEATURES].values
    y_test  = test[TARGET].values

    # ── Step 5: Scale features ───────────────────────────────────────────────
    # IMPORTANT: fit the scaler only on training data!
    # If we fit on the full dataset, test data statistics would influence
    # the scaling and that's a form of leakage.
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit + transform
    X_test_s  = scaler.transform(X_test)         # transform ONLY (no fit!)
    print("\n[4/7] Features scaled with StandardScaler (fit on train only)")

    # ── Step 6: Train ────────────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    print("\n[5/7] Model trained: LinearRegression")
    print(f"      Intercept: {model.intercept_:.4f}")

    # ── Step 7: Predict ──────────────────────────────────────────────────────
    y_train_pred = model.predict(X_train_s)
    y_test_pred  = model.predict(X_test_s)

    # ── Step 8: Evaluate ─────────────────────────────────────────────────────
    print("\n[6/7] Evaluation:")
    print("  --- Train ---")
    train_m = compute_metrics(y_train, y_train_pred, "Train")
    print("  --- Test ---")
    test_m  = compute_metrics(y_test, y_test_pred, "Test")

    # Overfitting check: if train and test RMSE are close, model generalises well
    rmse_gap = abs(test_m["rmse"] - train_m["rmse"])
    if rmse_gap < 5:
        print(f"\n  Overfitting check: RMSE gap = {rmse_gap:.2f} -- Good! Model generalises well.")
    else:
        print(f"\n  Overfitting check: RMSE gap = {rmse_gap:.2f} -- Consider adding regularization.")

    # ── Step 9: Plots ────────────────────────────────────────────────────────
    print("\n[7/7] Generating plots...")
    plot_actual_vs_predicted(train, test, y_train_pred, y_test_pred)
    plot_residuals(test, y_test_pred)
    plot_feature_importance(FEATURES, model.coef_)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Model         : LinearRegression")
    print(f"  No. of features: {len(FEATURES)}")
    print(f"  Train rows    : {len(train)}  |  Test rows: {len(test)}")
    print(f"  Test MAE      : {test_m['mae']:.2f} units/day")
    print(f"  Test RMSE     : {test_m['rmse']:.2f} units/day")
    print(f"  Test R2       : {test_m['r2']:.4f}")
    print("=" * 60)
    print("\n[DONE] Plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    main()
