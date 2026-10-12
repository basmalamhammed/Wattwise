"""Models: Linear Regression (forecast), K-Means (day patterns), and
Logistic Regression / KNN / Decision Tree / SVM (classification)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import features
from .features import FEATURES

RS = 42
LEVEL_NAMES = ["Low", "Medium", "High"]
TASK_LEVEL = "Level (Low/Medium/High)"
TASK_HIGH = "Unusually high (Yes/No)"


def split(ds: pd.DataFrame, frac: float = 0.8):
    """Chronological split (never shuffle time series)."""
    n = int(len(ds) * frac)
    return ds.iloc[:n], ds.iloc[n:]


# ---------------------------------------------------------------- regression
def _reg_row(name, y, p):
    return {
        "Model": name,
        "MAE (kWh)": mean_absolute_error(y, p),
        "RMSE (kWh)": float(np.sqrt(mean_squared_error(y, p))),
        "R²": r2_score(y, p),
    }


def fit_regression(train, test):
    lr = LinearRegression().fit(train[FEATURES], train["kwh"])
    pred = pd.Series(lr.predict(test[FEATURES]), index=test.index)
    table = pd.DataFrame(
        [
            _reg_row("Linear Regression", test["kwh"], pred),
            _reg_row("Baseline: same as yesterday", test["kwh"], test["lag1"]),
            _reg_row("Baseline: same weekday last week", test["kwh"], test["lag7"]),
        ]
    )
    return lr, table, pred


def forecast(daily: pd.DataFrame, lr: LinearRegression, days: int = 7) -> pd.Series:
    """Recursive multi-day forecast: each prediction feeds the next day's lag features."""
    hist = daily["kwh"].asfreq("D").interpolate(limit=3).dropna()
    out = {}
    for _ in range(days):
        d = hist.index[-1] + pd.Timedelta(days=1)
        row = pd.DataFrame([features.row_features(hist, d)])[FEATURES]
        yhat = max(float(lr.predict(row)[0]), 0.0)
        hist.loc[d] = yhat
        out[d] = yhat
    return pd.Series(out, name="forecast_kwh")


# ------------------------------------------------------------ classification
def _make_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=RS
        ),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", random_state=RS),
    }


def fit_classification(train, test):
    """Two tasks x four models. Level thresholds come from the TRAIN set only."""
    q = train["kwh"].quantile([1 / 3, 2 / 3]).to_numpy()
    targets = {
        TASK_LEVEL: (np.digitize(train["kwh"], q), np.digitize(test["kwh"], q), "macro"),
        TASK_HIGH: (train["high"].to_numpy(), test["high"].to_numpy(), "binary"),
    }
    rows, fitted = [], {}
    for task, (ytr, yte, avg) in targets.items():
        if len(np.unique(ytr)) < 2:  # nothing to learn from (e.g. no 'high' days in training)
            continue
        fitted[task] = {}
        for name, clf in _make_classifiers().items():
            pipe = make_pipeline(StandardScaler(), clf).fit(train[FEATURES], ytr)
            pred = pipe.predict(test[FEATURES])
            rows.append(
                {
                    "Task": task,
                    "Model": name,
                    "Accuracy": accuracy_score(yte, pred),
                    "Precision": precision_score(yte, pred, average=avg, zero_division=0),
                    "Recall": recall_score(yte, pred, average=avg, zero_division=0),
                    "F1": f1_score(yte, pred, average=avg, zero_division=0),
                }
            )
            fitted[task][name] = pipe
    cols = ["Task", "Model", "Accuracy", "Precision", "Recall", "F1"]
    return pd.DataFrame(rows, columns=cols), fitted, q


# ------------------------------------------------------------------- KNN days
def similar_days(ds: pd.DataFrame, date: pd.Timestamp, k: int = 5) -> pd.DataFrame:
    """KNN: the k past days whose pattern (features) is closest to `date`."""
    scaler = StandardScaler().fit(ds[FEATURES])
    X = scaler.transform(ds[FEATURES])
    pos = ds.index.get_loc(date)
    past = np.arange(len(ds)) < pos - 1
    nn = NearestNeighbors(n_neighbors=k).fit(X[past])
    _, idx = nn.kneighbors(X[[pos]])
    return ds.iloc[np.where(past)[0][idx[0]]][["kwh", "lag1", "roll7", "dow"]]


# ------------------------------------------------------------------- K-Means
def cluster_days(hourly: pd.DataFrame, k: int = 3):
    """Group complete days by usage pattern: total kWh + share used in each part of the day."""
    h = hourly["kwh"].to_frame()
    h["date"] = h.index.normalize()
    h["hour"] = h.index.hour
    pv = h.pivot(index="date", columns="hour", values="kwh").dropna()  # complete days only
    if len(pv) < 30:
        return None
    blocks = {
        "night": range(0, 6),
        "morning": range(6, 12),
        "afternoon": range(12, 18),
        "evening": range(18, 24),
    }
    prof = pd.DataFrame(index=pv.index)
    prof["total"] = pv.sum(axis=1)
    for name, hrs in blocks.items():
        prof[name] = pv[list(hrs)].sum(axis=1) / prof["total"]
    X = StandardScaler().fit_transform(prof)

    sil = {
        kk: silhouette_score(X, KMeans(n_clusters=kk, n_init=10, random_state=RS).fit(X).labels_)
        for kk in range(2, 7)
    }
    raw = pd.Series(KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X).labels_, index=prof.index)
    order = prof.groupby(raw)["total"].mean().sort_values().index
    rank = {c: i for i, c in enumerate(order)}
    prof["cluster"] = raw.map(rank)
    names = (
        {0: "Low-use days", 1: "Typical days", 2: "High-use days"}
        if k == 3
        else {i: f"Group {i + 1} (low→high)" for i in range(k)}
    )
    prof["label"] = prof["cluster"].map(names)
    prof["weekend"] = (prof.index.dayofweek >= 5).astype(int)
    curves = pv.groupby(prof["cluster"]).mean()
    curves.index = [names[i] for i in curves.index]
    return prof, sil, curves


# ----------------------------------------------------------- peaks and status
def hourly_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average kWh by day-of-week (rows 0=Mon) x hour (columns)."""
    h = hourly["kwh"].dropna().to_frame()
    h["hour"] = h.index.hour
    h["dow"] = h.index.dayofweek
    return h.pivot_table(index="dow", columns="hour", values="kwh", aggfunc="mean")


def consumption_status(daily: pd.DataFrame, window: int = 7, baseline: int = 30) -> dict:
    """Is recent consumption higher or lower than this household's usual?"""
    s = daily["kwh"].dropna()
    recent = s.iloc[-window:]
    base = s.iloc[-(window + baseline) : -window]
    base_mean, base_std = base.mean(), base.std()
    last = s.iloc[-1]
    change = (recent.mean() - base_mean) / base_mean * 100
    return {
        "recent_mean": recent.mean(),
        "baseline_mean": base_mean,
        "change_pct": change,
        "last_day": last,
        "last_date": s.index[-1],
        "last_day_unusual": bool(last > base_mean + base_std),
        "trend": "up" if change > 10 else "down" if change < -10 else "stable",
    }


# ----------------------------------------------------------------- everything
def fit_all(daily: pd.DataFrame, hourly: pd.DataFrame | None = None) -> dict:
    ds = features.build_dataset(daily)
    train, test = split(ds)
    lr, reg_table, reg_pred = fit_regression(train, test)
    clf_table, clf_fitted, level_q = fit_classification(train, test)
    return {
        "ds": ds,
        "train": train,
        "test": test,
        "lr": lr,
        "lr_full": LinearRegression().fit(ds[FEATURES], ds["kwh"]),  # for real forecasting
        "reg_table": reg_table,
        "reg_pred": reg_pred,
        "clf_table": clf_table,
        "clf_fitted": clf_fitted,
        "level_q": level_q,
        "clusters": cluster_days(hourly) if hourly is not None else None,
        "profile": hourly_profile(hourly) if hourly is not None else None,
    }
