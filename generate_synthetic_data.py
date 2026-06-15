"""
generate_synthetic_data.py

Creates a synthetic daily sales dataset for the sales-forecast-comparison project.
All data is randomly generated. It contains no real or proprietary information.

The data has realistic structure so the models have something to learn:
  - a gentle upward trend over time
  - a weekly pattern (weekdays busier than weekends)
  - a recurring vs non recurring split
  - random noise

Usage:
    python generate_synthetic_data.py
This writes daily_sales.csv to the current folder.
"""

import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)  # Reproducible: same data every run.

NUM_DAYS = 730            # about two years of daily data
START_DATE = datetime.now() - timedelta(days=NUM_DAYS)
OUTPUT_FILE = "daily_sales.csv"

# Weekday multipliers (Monday=0 ... Sunday=6): weekends are slower.
WEEKDAY_FACTOR = [1.05, 1.10, 1.10, 1.08, 1.00, 0.55, 0.45]


def build_rows(num_days):
    rows = []
    for day_index in range(num_days):
        date = START_DATE + timedelta(days=day_index)

        # Base level with a slow upward trend over time.
        base = 4000 + (day_index * 2.0)

        # Weekly seasonality.
        base *= WEEKDAY_FACTOR[date.weekday()]

        # A mild monthly wave using a sine curve.
        base *= 1 + 0.05 * math.sin(2 * math.pi * date.day / 30)

        # Split total into recurring and non recurring sales.
        recurring = base * random.uniform(0.45, 0.55)
        non_recurring = base - recurring

        # Add random noise so it is not perfectly predictable.
        noise = random.uniform(-250, 250)
        total = max(0, base + noise)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": date.strftime("%A"),
            "recurring_sales": round(recurring, 2),
            "non_recurring_sales": round(non_recurring, 2),
            "total_sales": round(total, 2),
        })
    return rows


def main():
    rows = build_rows(NUM_DAYS)
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
