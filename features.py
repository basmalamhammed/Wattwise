"""Feature engineering. Every feature for day t uses ONLY days before t (no leakage)."""
import pandas as pd

FEATURES = ["lag1", "lag2", "lag7", "roll7", "roll30", "dow", "is_weekend", "month"]


def build_dataset(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily kWh -> supervised dataset (features + target `kwh` + `high` label).

    `high` = 1 when the day's consumption is above (mean + 1 std) of the PREVIOUS 30 days,
    i.e. "unusually high compared with this household's own recent habit".
    """
    s = daily["kwh"].asfreq("D").interpolate(limit=3)
    prev = s.shift(1)
    df = pd.DataFrame({"kwh": s})
    df["lag1"] = s.shift(1)
    df["lag2"] = s.shift(2)
    df["lag7"] = s.shift(7)
    df["roll7"] = prev.rolling(7).mean()
    df["roll30"] = prev.rolling(30).mean()
    df["std30"] = prev.rolling(30).std()
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df.index.month
    df["high"] = (df["kwh"] > df["roll30"] + df["std30"]).astype(int)
    return df.dropna(subset=["kwh", "std30"] + FEATURES)


def row_features(history: pd.Series, date: pd.Timestamp) -> dict:
    """Features for `date`, given a continuous daily history that ends on date-1.
    Used for recursive multi-day forecasting."""
    return {
        "lag1": history.iloc[-1],
        "lag2": history.iloc[-2],
        "lag7": history.iloc[-7],
        "roll7": history.iloc[-7:].mean(),
        "roll30": history.iloc[-30:].mean(),
        "dow": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
        "month": date.month,
    }
