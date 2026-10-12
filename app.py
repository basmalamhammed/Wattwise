"""WattWise dashboard.   Run:  python -m streamlit run app.py"""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import data, models

st.set_page_config(page_title="WattWise", page_icon="⚡", layout="wide")

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SAMPLE = Path(__file__).parent / "samples" / "sample_hourly.csv"
MODE_DEMO, MODE_UPLOAD = "Demo data (UCI household)", "Upload my own CSV"


@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_demo():
    hourly, source = data.load_hourly()
    return hourly, data.to_daily(hourly), source


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    # sep=None lets pandas detect comma / semicolon / tab
    return pd.read_csv(BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")


@st.cache_resource(show_spinner="Training models...")
def train(_daily, _hourly, key):
    return models.fit_all(_daily, _hourly)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ WattWise")
st.sidebar.caption("Smart Electricity Consumption Analyzer")
mode = st.sidebar.radio("Data source", [MODE_DEMO, MODE_UPLOAD])

st.title("WattWise — Smart Electricity Consumption Analyzer")

if mode == MODE_DEMO:
    hourly, daily, source = load_demo()
    if source == "synthetic":
        st.error(
            "Showing **SYNTHETIC demo data**, not real readings. Put "
            "`household_power_consumption.txt` (UCI dataset) in the `data/` folder and reload."
        )
    else:
        st.caption(
            "Demo: UCI Individual Household Electric Power Consumption — one household in France "
            "(2006–2010). This is NOT your home."
        )
else:
    up = st.sidebar.file_uploader("Your readings (CSV)", type=["csv", "txt"])
    if up is None:
        st.info("Upload a CSV of your own electricity readings from the sidebar to analyse your home.")
        st.markdown(
            """
**What the file needs**
- One row per reading, with a **date/time** column and a **consumption in kWh** column
  (energy used during that reading, not the meter's running total).
- **Hourly (or finer) readings** unlock every tab. **Daily readings** work too, but peak-hour
  analysis and day patterns need hourly data.
- At least **120 days** of history (6+ months is better). Monthly bills are not enough.
- Where readings come from: a smart meter / your electricity company's app export, or a
  plug-in energy monitor (Shelly EM, Sonoff POW, Emporia, ...).

**Example**

| timestamp | kwh |
|---|---|
| 2025-01-01 00:00 | 0.42 |
| 2025-01-01 01:00 | 0.38 |
"""
        )
        if SAMPLE.exists():
            st.download_button(
                "Download a sample file (synthetic, for trying the upload)",
                SAMPLE.read_bytes(),
                file_name="sample_hourly.csv",
                mime="text/csv",
            )
        st.stop()

    try:
        raw_df = read_csv_bytes(up.getvalue())
    except Exception as e:  # unreadable / empty file
        st.error(f"Could not read this file as a table: {e}")
        st.stop()
    if raw_df.shape[1] < 2:
        st.error("The file needs at least two columns: a date/time and a consumption value.")
        st.stop()

    cols = list(raw_df.columns)
    g_ts, g_val = data.guess_columns(raw_df)
    ts_col = st.sidebar.selectbox("Date/time column", cols, index=cols.index(g_ts))
    val_col = st.sidebar.selectbox("Consumption (kWh) column", cols, index=cols.index(g_val) if g_val in cols else 0)
    time_opts = ["(none)"] + [c for c in cols if c not in (ts_col, val_col)]
    time_col = st.sidebar.selectbox("Separate time column (optional)", time_opts)
    dayfirst = st.sidebar.checkbox("Dates are day-first (31/12/2025)", value=True)
    st.sidebar.caption("Consumption must be kWh per reading. Readings are summed into hours and days.")
    with st.expander("Preview of your file"):
        st.dataframe(raw_df.head(10))

    try:
        hourly, daily, info = data.parse_upload(
            raw_df, ts_col, val_col, None if time_col == "(none)" else time_col, dayfirst
        )
    except (ValueError, KeyError, TypeError) as e:
        st.error(str(e))
        st.stop()
    source = "uploaded"
    st.success(
        f"Analysing your data: {info['days']} usable days "
        f"({daily['kwh'].dropna().index[0].date()} → {daily['kwh'].dropna().index[-1].date()}), "
        f"{info['granularity']} readings."
    )
    if info["dropped_rows"]:
        st.warning(f"{info['dropped_rows']} rows were skipped (unreadable date, empty or negative value).")
    if info["granularity"] == "daily":
        st.info("Daily readings: the 'Peak hours' and 'Day patterns' tabs need hourly readings.")
    if info["days"] < 365:
        st.warning("Less than a year of history: seasonal effects are not captured, so forecasts are rougher.")

key = (source, len(daily), str(daily.index[0]), str(daily.index[-1]), round(float(daily["kwh"].sum()), 3))
res = train(daily, hourly, key)

valid = daily["kwh"].dropna()
as_of = st.sidebar.date_input(
    "Analysis date",
    value=valid.index[-1].date(),
    min_value=(valid.index[0] + pd.Timedelta(days=60)).date(),
    max_value=valid.index[-1].date(),
)
horizon = st.sidebar.slider("Forecast horizon (days)", 3, 14, 7)
daily_asof = daily.loc[: pd.Timestamp(as_of)]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1 · Up or down?", "2 · Peak hours", "3 · Forecast", "4 · Day patterns", "5 · Model comparison"]
)

# ---------------------------------------------------------- 1. up or down
with tab1:
    st.subheader("Is my consumption higher or lower than usual?")
    stt = models.consumption_status(daily_asof)
    c1, c2, c3 = st.columns(3)
    c1.metric("Last 7 days (avg kWh/day)", f"{stt['recent_mean']:.1f}", f"{stt['change_pct']:+.1f}% vs usual")
    c2.metric("Usual (previous 30 days)", f"{stt['baseline_mean']:.1f}")
    c3.metric(f"Last day ({stt['last_date'].date()})", f"{stt['last_day']:.1f}")

    if stt["trend"] == "up":
        st.warning(f"Consumption is up {stt['change_pct']:.0f}% compared with your usual level.")
    elif stt["trend"] == "down":
        st.success(f"Consumption is down {abs(stt['change_pct']):.0f}% compared with your usual level.")
    else:
        st.info("Consumption is about the same as usual (within ±10%).")
    if stt["last_day_unusual"]:
        st.error("⚠️ Alert: the last day was unusually high compared with the previous 30 days.")

    last = daily_asof["kwh"].iloc[-90:]
    roll = daily_asof["kwh"].rolling(30).mean().iloc[-90:]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(last.index, last.values, label="Daily kWh", color="#1f77b4")
    ax.plot(roll.index, roll.values, label="30-day average", color="#ff7f0e", linewidth=2)
    ax.set_ylabel("kWh / day")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Days most similar to the latest day (KNN)"):
        ds = res["ds"]
        ds_asof = ds.loc[: pd.Timestamp(as_of)]
        if len(ds_asof) > 10:
            st.dataframe(models.similar_days(ds_asof, ds_asof.index[-1], k=5).round(2))

# ------------------------------------------------------------ 2. peak hours
with tab2:
    st.subheader("When do I use the most electricity?")
    prof = res["profile"]
    if prof is None:
        st.info("This tab needs hourly readings. Upload a file with hourly (or finer) data to see it.")
    else:
        overall = prof.mean()
        top3 = overall.sort_values(ascending=False).head(3).index.tolist()
        st.write("Highest-consumption hours on average: " + ", ".join(f"**{h:02d}:00**" for h in sorted(top3)))

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(overall.index, overall.values, color=["#d62728" if h in top3 else "#1f77b4" for h in overall.index])
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Average kWh")
            ax.set_xticks(range(0, 24, 3))
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            st.pyplot(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            im = ax.imshow(prof.values, aspect="auto", cmap="YlOrRd")
            ax.set_yticks(range(7), DOW)
            ax.set_xticks(range(0, 24, 3))
            ax.set_xlabel("Hour of day")
            fig.colorbar(im, ax=ax, label="Average kWh")
            fig.tight_layout()
            st.pyplot(fig)

# -------------------------------------------------------------- 3. forecast
with tab3:
    st.subheader("How much will I use in the coming days?")
    fc = models.forecast(daily_asof, res["lr_full"], horizon)
    hist = daily_asof["kwh"].iloc[-30:]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(hist.index, hist.values, label="Actual", color="#1f77b4")
    ax.plot(fc.index, fc.values, "o--", label="Forecast (Linear Regression)", color="#d62728")
    actual_future = daily["kwh"].reindex(fc.index).dropna()
    if len(actual_future):
        ax.plot(actual_future.index, actual_future.values, ".", color="black", label="Actual (known later)")
    ax.set_ylabel("kWh / day")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    c1, c2 = st.columns(2)
    c1.metric(f"Total next {horizon} days", f"{fc.sum():.0f} kWh")
    c2.metric("Average per day", f"{fc.mean():.1f} kWh")
    mae = res["reg_table"].iloc[0]["MAE (kWh)"]
    st.caption(
        f"Typical error of a one-day-ahead forecast on held-out data: ±{mae:.1f} kWh. "
        "Multi-day forecasts feed on their own predictions, so they get less reliable further out."
    )
    st.dataframe(fc.round(2).rename_axis("date").to_frame("forecast_kwh"))

# ------------------------------------------------------------ 4. K-Means
with tab4:
    st.subheader("What kinds of days do I have? (K-Means)")
    if res["clusters"] is None:
        st.info("This tab needs at least 30 complete days of hourly readings.")
    else:
        profile, sil, curves = res["clusters"]
        summary = profile.groupby("label").agg(
            days=("total", "size"), avg_kwh=("total", "mean"), weekend_share=("weekend", "mean")
        )
        st.dataframe(summary.round(2))
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for name, row in curves.iterrows():
            ax.plot(row.index, row.values, marker="o", markersize=3, label=name)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average kWh")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Silhouette score by number of clusters (higher = better separated): "
            + ", ".join(f"k={k}: {v:.2f}" for k, v in sil.items())
        )

# ------------------------------------------------------ 5. model comparison
with tab5:
    st.subheader("Model comparison (chronological 80/20 split)")
    st.markdown("**Forecasting daily kWh**")
    st.dataframe(res["reg_table"].round(3), hide_index=True)
    st.markdown("**Classification**")
    if res["clf_table"].empty:
        st.info("Not enough variety in the training days to train the classifiers.")
    for task in res["clf_table"]["Task"].unique():
        st.markdown(f"*{task}*")
        st.dataframe(
            res["clf_table"][res["clf_table"]["Task"] == task].drop(columns="Task").round(3),
            hide_index=True,
        )
    st.caption(
        "'Unusually high' = a day above the mean + 1 standard deviation of the previous 30 days. "
        "It is a rare class, so compare F1 (not just accuracy)."
    )
    st.info(
        "Scope note: models are trained on the data source selected in the sidebar, so they describe "
        "that household only. They do not promise savings for any other home."
    )
