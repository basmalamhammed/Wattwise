# WattWise — Smart Electricity Consumption Analyzer

WattWise is a machine learning project that analyzes household electricity consumption patterns, identifies unusual increases in daily usage, discovers peak consumption periods, and forecasts future electricity consumption.

The project uses historical electricity readings from a single household to demonstrate a complete time-series analysis and machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and interactive visualization.

## Key Features

- **Consumption Analysis:** Determine whether daily electricity consumption has increased or decreased compared with recent historical patterns.
- **Peak Usage Detection:** Identify high-consumption periods and explore daily and weekly usage patterns.
- **Consumption Forecasting:** Predict next-day electricity consumption and generate multi-day forecasts.
- **Anomaly Detection:** Identify days with unusually high consumption relative to their preceding 30-day history.
- **Consumption Level Classification:** Categorize daily consumption as Low, Medium, or High.
- **Usage Pattern Clustering:** Group days with similar consumption characteristics using K-Means.
- **Interactive Dashboard:** Explore results through a Streamlit dashboard.
- **Custom CSV Upload:** Analyze a user's own electricity consumption data by retraining the models on the uploaded dataset.

> **Important:** The original dataset represents a single household. WattWise is an analytical and educational project, not a validated system for guaranteeing electricity savings or generalizing consumption patterns to every household. Predictions and classifications depend on the quality and characteristics of the available data.

## Dataset

WattWise uses the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from the UCI Machine Learning Repository.

The dataset contains measurements collected at one-minute intervals from a single household. The project primarily uses the `Global_active_power` feature, measured in kilowatts (kW), to derive hourly and daily consumption statistics.

The original dataset contains approximately two million minute-level records.

### Data Source Setup

1. Download the dataset from the UCI Machine Learning Repository.
2. Extract the downloaded archive.
3. Place `household_power_consumption.txt` in the project's `data/` directory.

If the original dataset is unavailable, WattWise automatically falls back to synthetic demonstration data and displays a warning in the dashboard. **Synthetic data is intended for testing the application only; results obtained from it must not be presented as real-world experimental findings.**

## Installation and Usage

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wattwise
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Machine Learning Pipeline

```bash
python train.py
```

This command executes the training and evaluation pipeline and prints the results in the terminal.

### 4. Launch the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to interact with the dashboard.

### Data Processing and Caching

On the first run with the original dataset, preprocessing may take some time because of the large number of minute-level readings.

Processed hourly data is cached in `data/hourly_cache.csv` to accelerate subsequent runs. If the original dataset changes, delete the cached file to regenerate the processed data.

## Analyze Your Own Electricity Data

WattWise supports custom datasets through the **Upload my own CSV** option in the dashboard sidebar.

When a CSV file is uploaded, the application preprocesses the data and retrains the models using the uploaded household's readings rather than the original UCI dataset.

### Required CSV Format

The dataset must contain one row per observation, including a timestamp and the electricity consumption for the corresponding measurement interval.

| timestamp | kwh |
|---|---:|
| 2025-01-01 00:00 | 0.42 |
| 2025-01-01 01:00 | 0.38 |
| 2025-01-01 02:00 | 0.35 |

- `timestamp`: The date and time of the observation.
- `kwh`: Electricity consumed during the measurement interval, in kilowatt-hours (kWh). This must be interval consumption, not a cumulative electricity meter reading.

### Supported Data Frequencies

- **Hourly or sub-hourly readings:** Supported. Readings finer than one hour are aggregated by summing the interval consumption within each hour.
- **Daily readings:** Supported for daily analysis, forecasting, and compatible classification tasks. Peak-hour analysis and time-of-day usage patterns require hourly or finer-resolution data.

### Data Requirements

- **Minimum history:** 120 days.
- **Recommended history:** At least six months; one full year is preferable for capturing seasonal patterns.
- Monthly electricity bills alone are insufficient for the project's time-series analysis.

The application automatically detects common CSV delimiters, including commas, semicolons, and tabs. Decimal commas, such as `0,42`, are also supported. Timestamps may be stored in a single column or split across separate date and time columns.

For dates in `DD/MM/YYYY` format, keep the day-first parsing option enabled. Dates in `YYYY-MM-DD` format are also supported.

A synthetic example file is provided at `samples/sample_hourly.csv`. It is intended exclusively for testing the upload functionality.

### Obtaining Your Electricity Readings

Depending on local availability and metering capabilities, readings may be obtained from:

- Electricity provider applications or smart-meter exports.
- Compatible home energy monitoring devices.
- Devices such as Shelly EM, Sonoff POW, or Emporia energy monitors.

Device compatibility and data export availability depend on the specific model and installation.

## Project Architecture

```text
wattwise/
├── app.py                         # Streamlit dashboard
├── train.py                       # Training and evaluation entry point
├── requirements.txt               # Project dependencies
├── data/
│   └── household_power_consumption.txt
├── samples/
│   └── sample_hourly.csv          # Synthetic sample data
└── src/
    ├── data.py                    # Loading, cleaning, and aggregation
    ├── features.py                # Feature engineering and target labels
    └── models.py                  # Model training and evaluation
```

## Machine Learning Workflow

WattWise follows a time-series machine learning workflow designed to preserve chronological order and reduce data leakage.

### 1. Data Loading and Preprocessing

- Load the original minute-level electricity measurements.
- Convert missing-value markers (`?`) to `NaN`.
- Aggregate valid minute-level readings into hourly consumption values.
- Treat hours with fewer than 30 valid minutes as missing.
- Aggregate hourly readings into daily consumption totals.
- Exclude days with fewer than 20 valid hours.

For the original dataset, `Global_active_power` is measured in kW. Since the source readings are taken at one-minute intervals, hourly energy consumption is derived from the power measurements and expressed in kWh. Under the project's aggregation rules, the numerical hourly mean power in kW corresponds approximately to the hourly energy consumed in kWh.

### 2. Feature Engineering

The forecasting and classification models use historical consumption and calendar information, including:

- Previous-day consumption (lag 1).
- Consumption from two days earlier (lag 2).
- Consumption from the same day of the previous week (lag 7).
- Rolling averages over the previous 7 and 30 days.
- Day of the week.
- Month of the year.

For a target day \(t\), historical features are constructed using information available before day \(t\). Rolling statistics are shifted appropriately to avoid incorporating the target day's consumption into its own predictive features.

### 3. Chronological Train-Test Split

The dataset is split chronologically into training and testing subsets using an 80/20 ratio.

Random shuffling is avoided because electricity consumption is a time series. The test set therefore represents a later period than the training set.

### 4. Model Evaluation

Forecasting models are compared with simple persistence baselines:

- **Previous-day baseline:** Predict consumption using the previous day's value.
- **Previous-week baseline:** Predict consumption using the value from the same weekday in the previous week.

These baselines help determine whether machine learning models provide meaningful improvements over simple historical predictions.

## Machine Learning Algorithms

| Algorithm | Task | Purpose |
|---|---|---|
| Linear Regression | Regression | Predict next-day consumption and generate recursive multi-day forecasts. |
| K-Nearest Neighbors (KNN) | Classification and historical similarity analysis | Compare consumption patterns with similar historical days and provide a classification baseline. |
| Decision Tree | Classification | Categorize daily consumption into Low, Medium, and High levels. |
| Logistic Regression | Binary classification | Identify days with unusually high consumption relative to recent history. |
| Support Vector Machine (SVM, RBF kernel) | Classification | Compare classification performance using a nonlinear decision boundary. |
| K-Means | Unsupervised learning | Group days according to total consumption and its distribution across parts of the day. |

## Target Definitions

### Consumption Levels: Low, Medium, and High

The classification thresholds are based on the 33rd and 67th percentiles (terciles) of the training-period consumption values.

The thresholds are calculated using training data only and then applied consistently to the relevant evaluation data.

### Unusually High Consumption

A day is labeled as unusually high when its consumption exceeds the historical mean by more than one standard deviation, calculated from the preceding 30 days:

\[
E_t > \mu_{t-30:t-1} + \sigma_{t-30:t-1}
\]

Where:

- \(E_t\) is the consumption on the target day.
- \(\mu_{t-30:t-1}\) is the mean consumption over the preceding 30 days.
- \(\sigma_{t-30:t-1}\) is the standard deviation over the same historical window.

The target day is excluded from the calculation to prevent information leakage.

Because unusually high-consumption days may be relatively rare, classification performance should not be judged by accuracy alone. **F1-score and recall** are particularly relevant for evaluating the detection of these events.

## Technologies

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Dashboard:** Streamlit
- **Data Visualization:** Based on the visualization libraries configured in the project dependencies
- **Data Source:** UCI Machine Learning Repository

## Project Goals

WattWise was developed as a practical machine learning project to demonstrate:

- End-to-end data preprocessing and feature engineering.
- Time-series forecasting and chronological evaluation.
- Supervised and unsupervised learning.
- Baseline comparisons and appropriate evaluation metrics.
- Leakage-aware model development.
- Interactive data exploration and custom dataset support.

## Limitations

- The original dataset represents one household and may not generalize to other households.
- Forecast quality depends on data completeness, historical consistency, and consumption patterns.
- Multi-day forecasts generated recursively may accumulate prediction errors.
- Synthetic fallback data is not evidence of real-world predictive performance.
- The project identifies statistical deviations; an unusual increase does not necessarily indicate a fault, waste, or a specific appliance.
- WattWise does not directly control electrical appliances or guarantee reductions in electricity bills.

## Future Improvements

Potential extensions include:

- Evaluating additional forecasting models against the existing baselines.
- Adding uncertainty intervals to multi-day forecasts.
- Improving robustness to missing readings and changes in household behavior.
- Evaluating performance across multiple households.
- Adding explainability features to clarify why a day was classified as unusual.
- Measuring model performance across different seasons and consumption regimes.

## License

Add a `LICENSE` file to specify the terms under which this project may be used, modified, and distributed.

---

**WattWise — Turning household electricity readings into interpretable consumption patterns and forecasts.**

### Layout smoke tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

These checks cover the package layout, bundled sample location, and labelled
synthetic fallback. Running `python train.py` without the UCI data is a synthetic
smoke test only, not a reproduction of any real-dataset results.
