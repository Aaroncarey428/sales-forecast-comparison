"""
forecast_comparison.py

Compares two approaches to forecasting daily sales:
  1. A run rate baseline (trailing average projected forward)
  2. A trained machine learning regression model (scikit-learn)

Both are evaluated on the same held out test data using standard metrics,
so the comparison is fair and honest.

All data is synthetic. No real or proprietary information is used.

Usage:
    python generate_synthetic_data.py   (first, to create daily_sales.csv)
    python forecast_comparison.py
"""

import csv
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE = "daily_sales.csv"
OUTPUT_DIR = "sample_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.csv")


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
#    Turn raw daily sales into inputs a model can learn from.
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Calendar features.
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week_num"] >= 5).astype(int)

    # Lag features: sales from previous days.
    for lag in (1, 7):
        df[f"lag_{lag}"] = df["total_sales"].shift(lag)

    # Rolling averages over recent windows.
    for window in (7, 30):
        df[f"rolling_mean_{window}"] = (
            df["total_sales"].shift(1).rolling(window).mean()
        )

    # Share of recurring activity, a domain specific signal.
    df["recurring_share"] = df["recurring_sales"] / df["total_sales"]

    # Drop the early rows that have no lag or rolling history yet.
    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Split into training and test sets
#    Time series: train on the earlier portion, test on the most recent.
# ---------------------------------------------------------------------------
def time_split(df, test_fraction=0.2):
    split_point = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    return train, test


# ---------------------------------------------------------------------------
# 4. Baseline model: run rate
#    Predict each day as the trailing 7 day average (the run rate idea).
# ---------------------------------------------------------------------------
def run_rate_predictions(test):
    # rolling_mean_7 is already the trailing 7 day average up to the prior day.
    return test["rolling_mean_7"].values


# ---------------------------------------------------------------------------
# 5. Trained model: random forest regression
# ---------------------------------------------------------------------------
FEATURES = [
    "day_of_week_num", "day_of_month", "month", "is_weekend",
    "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_30", "recurring_share",
]


def train_model(train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[FEATURES], train["total_sales"])
    return model


# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate(name, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = math_sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"model": name, "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "MAPE_percent": round(mape, 2)}


def math_sqrt(value):
    return value ** 0.5


# ---------------------------------------------------------------------------
# 7. Run the full comparison
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(
            f"{INPUT_FILE} not found. Run: python generate_synthetic_data.py first."
        )

    df = load_data(INPUT_FILE)
    df = engineer_features(df)
    train, test = time_split(df)

    # Baseline predictions.
    baseline_pred = run_rate_predictions(test)
    baseline_result = evaluate("Run rate baseline",
                               test["total_sales"], baseline_pred)

    # Trained model predictions.
    model = train_model(train)
    model_pred = model.predict(test[FEATURES])
    model_result = evaluate("Trained regression",
                            test["total_sales"], model_pred)

    results = [baseline_result, model_result]

    # Print a readable comparison.
    print("\nForecast comparison results (lower is better):\n")
    print(f"{'Model':<22}{'MAE':>12}{'RMSE':>12}{'MAPE %':>12}")
    for r in results:
        print(f"{r['model']:<22}{r['MAE']:>12}{r['RMSE']:>12}{r['MAPE_percent']:>12}")

    # Save results to CSV.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
