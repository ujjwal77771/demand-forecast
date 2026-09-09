"""
generate_data.py
================
Author : Ujjwal Deep
Project: Simple Demand Forecast (ML Basics - Day 1)

What this file does:
--------------------
Since I don't have a real dataset, I'm generating fake (synthetic) data
that looks realistic. The demand has:
  1. An upward TREND  → the product gets more popular over time
  2. WEEKEND SPIKES   → people shop more on Sat/Sun
  3. SUMMER PEAK      → demand is higher in summer (Jun-Aug)
  4. RANDOM NOISE     → real data is never perfectly smooth

Output: data/demand_data.csv  (730 rows = 2 years of daily data)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_demand(start_date="2022-01-01", periods=730, seed=42):
    """
    Generate synthetic daily demand data.

    Parameters
    ----------
    start_date : str   - first date in the dataset
    periods    : int   - number of days to generate
    seed       : int   - random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: ['date', 'demand']
    """

    # Fix the random seed so results are the same every run
    rng = np.random.default_rng(seed)

    # Create a range of dates (one per day)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    # --- Component 1: Trend (demand slowly grows from 100 to 150) -----------
    trend = np.linspace(100, 150, periods)

    # --- Component 2: Weekend spike (+30 units on Sat=5 and Sun=6) ----------
    # .to_numpy() is needed to get a plain array (not a pandas Index)
    weekly = np.where(dates.dayofweek.to_numpy() >= 5, 30.0, 0.0)

    # --- Component 3: Summer peak using a sine wave -------------------------
    # dayofyear goes from 1 to 365; peak is around day 172 (mid June)
    day_of_year = dates.dayofyear.to_numpy().astype(float)
    monthly = 20.0 * np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.0)

    # --- Component 4: Random noise (mean=0, std=12) -------------------------
    noise = rng.normal(0.0, 12.0, periods)

    # Add all components together, clip at 0 (demand can't be negative)
    demand = np.clip(trend + weekly + monthly + noise, 0, None)
    demand = demand.round().astype(int)

    df = pd.DataFrame({"date": dates, "demand": demand})
    return df


def main():
    # Save the data one folder up from src/, inside data/
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demand_data.csv"

    df = generate_demand()
    df.to_csv(out_path, index=False)

    print("[OK] Dataset saved ->", out_path)
    print("     Shape :", df.shape)
    print("     Period:", df["date"].min().date(), "->", df["date"].max().date())
    print("     Demand  min:", df["demand"].min(),
          " mean:", round(df["demand"].mean(), 1),
          " max:", df["demand"].max())


if __name__ == "__main__":
    main()
