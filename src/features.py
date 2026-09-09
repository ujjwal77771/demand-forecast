"""
features.py
===========
Author : Ujjwal Deep
Project: Simple Demand Forecast (ML Basics - Day 1)

What this file does:
--------------------
Raw data only has 'date' and 'demand'. A model can't learn from a date string!
So we create NEW columns (features) that the model CAN learn from:

  Calendar features  → tell the model what time of year/week it is
  Lag features       → tell the model what demand looked like in the past
  Rolling mean       → smoothed recent demand level

Why lag features?
-----------------
If demand was high last week (lag_7), it's likely high this week too.
This is the key insight in time-series forecasting.

Why rolling mean?
-----------------
Averages out the noise in recent demand, giving the model a cleaner signal.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df):
    """
    Add time-based features extracted from the 'date' column.

    New columns added:
      trend        - integer index (0, 1, 2, ...) - captures long-term growth
      day_of_week  - 0=Monday, 6=Sunday
      month        - 1 to 12
      week_of_year - 1 to 52
      is_weekend   - 1 if Saturday or Sunday, else 0
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date (important - always work chronologically)
    df = df.sort_values("date").reset_index(drop=True)

    df["trend"]        = np.arange(len(df))                     # 0, 1, 2, 3...
    df["day_of_week"]  = df["date"].dt.dayofweek                # 0=Mon, 6=Sun
    df["month"]        = df["date"].dt.month                    # 1-12
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)   # 1 if Sat/Sun

    return df


def add_lag_features(df, target="demand"):
    """
    Add lag and rolling window features.

    IMPORTANT NOTE about lag features:
    -----------------------------------
    We use .shift() to look BACKWARDS in time.
    shift(7) means "use the value from 7 days ago".
    This avoids 'data leakage' — the model must NOT see future demand.

    Rolling mean uses shift(1) first so today's value is not included.

    The first 14 rows will have NaN (no data before them) — we drop those later.

    New columns added:
      lag_7          - demand from 7 days ago (same day last week)
      lag_14         - demand from 14 days ago
      rolling_mean_7 - average demand over the previous 7 days
    """
    df = df.copy()
    df["lag_7"]          = df[target].shift(7)
    df["lag_14"]         = df[target].shift(14)
    # shift(1) first so we don't include today in the rolling window (leakage!)
    df["rolling_mean_7"] = df[target].shift(1).rolling(window=7).mean()
    return df


def get_feature_columns():
    """
    Return the list of feature columns used for model training.
    The order matches how the scaler and model were fitted.
    """
    return [
        "trend",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
    ]
