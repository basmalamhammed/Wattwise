"""Data loading and preprocessing for WattWise.

Real data: UCI "Individual household electric power consumption"
(household_power_consumption.txt, minute readings, ';' separated, '?' = missing).
Put the file in  wattwise/data/  and it is used automatically.

If the file is not there, a clearly-labelled SYNTHETIC dataset is generated so the
pipeline and dashboard can still be run and tested.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "household_power_consumption.txt"
CACHE_FILE = DATA_DIR / "hourly_cache.csv"


def _parse_raw(path: Path) -> pd.DataFrame:
    """Minute readings (kW) -> hourly energy (kWh).

    Global_active_power is in kW. The mean kW over an hour equals the kWh used in
    that hour. Hours with fewer than 30 valid minutes are treated as missing.
    """
    df = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?", ""],
        low_memory=False,
    )
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    kw = pd.Series(pd.to_numeric(df["Global_active_power"], errors="coerce").values, index=ts)
    g = kw.resample("h").agg(["mean", "count"])
    kwh = g["mean"].where(g["count"] >= 30)
    return kwh.to_frame("kwh")


def _synthetic(days: int = 1100, seed: int = 7) -> pd.DataFrame:
    """Synthetic hourly data (morning + evening peaks, winter bump, weekends, absences)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-01-01", periods=days * 24, freq="h")
    hour, doy, dow = idx.hour.values, idx.dayofyear.values, idx.dayofweek.values
    profile = (
        0.35
        + 0.9 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        + 1.3 * np.exp(-0.5 * ((hour - 20) / 2.0) ** 2)
    )
    season = 1 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365)
    weekend = np.where(dow >= 5, 1.15, 1.0)
    day_factor = rng.lognormal(0, 0.15, size=days).repeat(24)
    noise = rng.lognormal(0, 0.2, size=len(idx))
    kwh = profile * season * weekend * day_factor * noise
    for start in rng.choice(days - 10, 6, replace=False):  # travel / absence periods
        kwh[start * 24 : (start + int(rng.integers(3, 9))) * 24] *= 0.35
    return pd.DataFrame({"kwh": kwh}, index=idx)


def load_hourly() -> tuple[pd.DataFrame, str]:
    """Return (hourly DataFrame with column 'kwh', source) where source is 'UCI' or 'synthetic'."""
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True), "UCI"
    if RAW_FILE.exists():
        hourly = _parse_raw(RAW_FILE)
        hourly.to_csv(CACHE_FILE)
        return hourly, "UCI"
    return _synthetic(), "synthetic"


def to_daily(hourly: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """Hourly kWh -> daily kWh. Days with fewer than `min_hours` valid hours are dropped;
    days with a few missing hours are scaled up to a full 24 h."""
    g = hourly["kwh"].resample("D").agg(["sum", "count"])
    kwh = (g["sum"] * 24 / g["count"]).where(g["count"] >= min_hours)
    return kwh.to_frame("kwh")


# --------------------------------------------------------------- user uploads
MIN_DAYS = 120  # below this there is not enough history to train and test


def guess_columns(df: pd.DataFrame):
    """Guess (timestamp column, consumption column) from the column names."""
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    ts = next(
        (c for c in cols if any(k in low[c] for k in ("date", "time", "stamp", "تاريخ"))), cols[0]
    )
    val = next(
        (
            c
            for c in cols
            if c != ts and any(k in low[c] for k in ("kwh", "energy", "consum", "usage", "استهلاك"))
        ),
        None,
    )
    if val is None:
        numeric = [c for c in cols if c != ts and pd.api.types.is_numeric_dtype(df[c])]
        val = numeric[0] if numeric else next((c for c in cols if c != ts), None)
    return ts, val


def parse_upload(df, ts_col, val_col, time_col=None, dayfirst=True):
    """Turn a user's table into (hourly or None, daily, info).

    The consumption column must be energy per reading in kWh (readings are SUMMED when
    grouped into hours/days). Hourly-or-finer files give hourly + daily results; daily
    files give daily only (peak-hour analysis is then unavailable).
    """
    ts_text = df[ts_col].astype(str)
    if time_col:
        ts_text = ts_text + " " + df[time_col].astype(str)
    # Year-first text (2025-03-04) is unambiguous; dayfirst would wrongly swap month and day.
    if ts_text.str.match(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}").mean() > 0.9:
        dayfirst = False
    ts = pd.to_datetime(ts_text, dayfirst=dayfirst, errors="coerce")
    vals = pd.to_numeric(df[val_col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    vals = vals.where(vals >= 0)  # negative readings are invalid
    ok = ts.notna() & vals.notna()
    dropped = int((~ok).sum())
    if ok.sum() < 48:
        raise ValueError(
            f"Only {int(ok.sum())} valid rows could be read. Check the date column, the consumption "
            "column and the 'day first' option. (Monthly bills have too few readings to analyse.)"
        )
    s = pd.Series(vals[ok].to_numpy(), index=pd.DatetimeIndex(ts[ok])).sort_index()
    s = s[~s.index.duplicated(keep="first")]

    step = s.index.to_series().diff().median()
    if step >= pd.Timedelta(hours=20):
        hourly = None
        daily = s.resample("D").sum(min_count=1).to_frame("kwh")
        granularity = "daily"
    else:
        hourly = s.resample("h").sum(min_count=1).to_frame("kwh")
        daily = to_daily(hourly)
        granularity = "hourly"

    n_days = int(daily["kwh"].notna().sum())
    if n_days < MIN_DAYS:
        raise ValueError(
            f"Only {n_days} usable days found; at least {MIN_DAYS} are needed "
            "(daily or hourly readings, a few months or more). Monthly bills are not enough."
        )
    return hourly, daily, {"granularity": granularity, "dropped_rows": dropped, "days": n_days}
