"""Run the whole ML pipeline from the command line and print the results.

    python train.py
"""
from pathlib import Path

import pandas as pd

from src import data, models

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")


def main():
    hourly, source = data.load_hourly()
    daily = data.to_daily(hourly)
    print(f"Data source : {source}")
    print(f"Hourly rows : {len(hourly):,}   Valid days: {daily['kwh'].notna().sum():,}")
    print(f"Date range  : {daily.index.min().date()} -> {daily.index.max().date()}\n")

    res = models.fit_all(daily, hourly)
    print(f"Train days: {len(res['train'])}   Test days: {len(res['test'])}\n")

    print("== Forecasting next-day consumption (test set) ==")
    print(res["reg_table"].to_string(index=False), "\n")

    print("== Classification (test set) ==")
    print(res["clf_table"].to_string(index=False), "\n")

    prof, sil, _ = res["clusters"]
    print("== K-Means day patterns (k=3) ==")
    print(prof.groupby("label").agg(days=("total", "size"), avg_kwh=("total", "mean"),
                                    weekend_share=("weekend", "mean")).to_string())
    print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "\n")

    print("== Next 7 days forecast ==")
    print(models.forecast(daily, res["lr_full"], 7).round(2).to_string())

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    res["reg_table"].to_csv(out / "regression_metrics.csv", index=False)
    res["clf_table"].to_csv(out / "classification_metrics.csv", index=False)
    print(f"\nMetrics saved to {out}/")


if __name__ == "__main__":
    main()
