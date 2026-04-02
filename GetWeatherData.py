from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str


MONTREAL = Location(
    name="Montreal, QC",
    lat=45.5017,
    lon=-73.5673,
    timezone="America/Montreal",
)

def fetch_open_meteo_hourly(
    start_date: str,
    end_date: str,
    location: Location = MONTREAL,
    hourly_vars: List[str] | None = None,
) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "cloud_cover",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": location.timezone,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", None)
    if times is None:
        raise RuntimeError(f"Open-Meteo response missing 'hourly.time'. Keys: {payload.keys()}")

    idx = pd.to_datetime(times)
    df = pd.DataFrame(index=idx)
    for k, v in hourly.items():
        if k == "time":
            continue
        df[k] = v
    df.index.name = "time_local"
    return df

# Preprocessing + features
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    df = df.reindex(full_idx)

    df = df.interpolate(limit=6)
    df = df.ffill().bfill()

    rename = {
        "temperature_2m": "T",
        "wind_speed_10m": "W",
        "wind_direction_10m": "Wd",
        "relative_humidity_2m": "RH",
        "surface_pressure": "P",
        "precipitation": "Prec",
        "cloud_cover": "Cloud",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Diurnal + seasonal features
    hour = df.index.hour.to_numpy()
    omega = 2 * math.pi / 24.0
    
    # daily cycle terms 
    df["sin_day"] = np.sin(omega * hour)
    df["cos_day"] = np.cos(omega * hour)

    doy = df.index.dayofyear.to_numpy()
    omega_y = 2 * math.pi / 365.25

    # yearly (seasonal) cycle terms
    df["sin_year"] = np.sin(omega_y * doy)
    df["cos_year"] = np.cos(omega_y * doy)

    return df

def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for L in lags:
        if L <= 0:
            continue
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

def split_train_val(data: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) <= val_hours + 10:
        raise ValueError("Not enough samples for requested validation window.")
    return data.iloc[:-val_hours].copy(), data.iloc[-val_hours:].copy()

def build_prediction_model(train: pd.DataFrame, val: pd.DataFrame, features: list, target_column: str) -> tuple[np.ndarray, np.ndarray]:
    # construct design matrix (for training)
    X_train = train[features].to_numpy()
    
    # target values for temperature 
    y_train = train[target_column].to_numpy()

    X_val = val[features].to_numpy()    # design matrix (for validation)
    y_val = val[target_column].to_numpy()    # true values for validation

    theta, _, _, _ = np.linalg.lstsq(X_train, y_train)    # compute theta
    y_pred = X_val @ theta  # prediction equation

    return y_pred, y_val    

def compute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Return (RMSE, MAE) for a pair of arrays.
    """
    residuals = y_true - y_pred
    return float(np.sqrt(np.mean(residuals ** 2))), float(np.mean(np.abs(residuals)))

def print_header(title: str) -> None:
    """
    Print a section header to the console.
    """
    bar = "=" * 51
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def print_error_table(
    horizons: List[int],
    model_errors: List[Tuple[float, float]],
    baseline_errors: List[Tuple[float, float]] | None,
    variable: str,
) -> None:
    """
    Print a formatted RMSE / MAE table for one parameter across horizons.
    """
    col = 14
    print(f"\n  {variable} errors")
    if baseline_errors:
        print(f"  {'Horizon':>8}  {'Model RMSE':>{col}}  {'Model MAE':>{col}}  {'Base RMSE':>{col}}  {'Base MAE':>{col}}  {'ΔRMSE %':>8}  {'ΔMAE %':>8}")
        for h, (rmse_m, mae_m), (rmse_b, mae_b) in zip(horizons, model_errors, baseline_errors):
            pct_rmse = 100 * (rmse_b - rmse_m) / rmse_b
            pct_mae  = 100 * (mae_b  - mae_m)  / mae_b
            print(f"  {h:>8}  {rmse_m:>{col}.4f}  {mae_m:>{col}.4f}  {rmse_b:>{col}.4f}  {mae_b:>{col}.4f}  {pct_rmse:>7.1f}%  {pct_mae:>7.1f}%")
    else:
        print(f"  {'Horizon':>8}  {'RMSE':>{col}}  {'MAE':>{col}}")
        for h, (rmse, mae) in zip(horizons, model_errors):
            print(f"  {h:>8}  {rmse:>{col}.4f}  {mae:>{col}.4f}")

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    ylabel: str,
    baseline: np.ndarray | None = None,
) -> None:
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted" if baseline is None else "Model")
    if baseline is not None:
        plt.plot(baseline, label="Baseline")
    plt.xlabel("Time (hours)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ===================================================
    # DATE, FEATURE, AND HORIZON SET UP
    # =================================================== 
    start_date = "2025-01-01"
    end_date = "2025-12-31"

    start_date_summer = "2024-06-01"
    end_date_summer = "2025-08-31"

    start_date_winter = "2024-12-01"
    end_date_winter = "2026-02-28"
    
    montreal = Location(
        name="Montreal, QC",
        lat=45.5017,
        lon=-73.5673,
        timezone="America/Montreal",
    )

    print(f"Fetching Open-Meteo hourly data for {montreal.name}...")
    df_raw = fetch_open_meteo_hourly(start_date, end_date, location=montreal)
    print("Preprocessing...")
    df = preprocess(df_raw)
    print(f"Data between {start_date} and {end_date} is used.")
    
    df = add_lags(df, "T", [1, 2, 3, 6, 12, 24])
    df = add_lags(df, "W", [1, 3, 6, 12])

    horizon = [1, 3, 6, 12, 24, 48]    # number of hours to predict into future

    val_hours = 92 * 24  # always predict last 3 months

    # Features vector
    features = [
    "T", "W",
    "T_lag1", "T_lag2", "T_lag3", "T_lag6", "T_lag12", "T_lag24",
    "W_lag1", "W_lag3", "W_lag6", "W_lag12",
    "sin_day", "cos_day",
    "sin_year", "cos_year",
    ]
    additional_features = ["Wd", "RH", "P", "Prec", "Cloud"]

    # ===================================================
    # SECTION 7.1 — TEMPERATURE MODEL
    # ===================================================
    print_header("SECTION 7.1 — TEMPERATURE MODEL")

    T_model_errors = []
    for h in horizon:
        df["target_T"] = df["T"].shift(-h)
        df_model = df.dropna()  # remove data with missing values
        train, val = split_train_val(df_model, val_hours)
        y_pred_T, y_val_T = build_prediction_model(train, val, features, "target_T")
        T_model_errors.append(compute_errors(y_val_T, y_pred_T))
        plot_actual_vs_predicted(y_val_T, y_pred_T, f"Validation: Actual vs Predicted Temperature with h = {h}", "Temperature (°C)")
    
    print_error_table(horizon, T_model_errors, None, "Temperature (°C)")


    # ===================================================
    # SECTION 7.2 — WIND SPEED MODEL
    # ===================================================
    print_header("SECTION 7.2 — WIND SPEED MODEL")

    W_model_errors = []
    for h in horizon:
        df["target_W"] = df["W"].shift(-h)
        df_model = df.dropna()  # remove data with missing values
        train, val = split_train_val(df_model, val_hours)
        y_pred_W, y_val_W = build_prediction_model(train, val, features, "target_W")
        W_model_errors.append(compute_errors(y_val_W, y_pred_W))
        plot_actual_vs_predicted(y_val_W, y_pred_W, f"Validation: Actual vs Predicted Wind Speed with h = {h}", "Wind Speed (km/h)")
    
    print_error_table(horizon, W_model_errors, None, "Wind Speed (km/h)")

    # ===================================================
    # SECTION 7.3 — TEMPERATURE BASELINE MODEL 
    # ===================================================
    print_header("SECTION 7.3 — TEMPERATURE BASELINE MODEL")

    T_baseline_errors = []
    for h in horizon:
        df_h = df.copy()
        df_h["target_T"] = df_h["T"].shift(-h)
        df_h = df_h.dropna()
        train, val = split_train_val(df_h, val_hours)
        y_pred_T, y_val_T = build_prediction_model(train, val, features, "target_T")
        baseline_T = val["T"].to_numpy()
        T_baseline_errors.append(compute_errors(y_val_T, baseline_T))
        plot_actual_vs_predicted(y_val_T, y_pred_T, f"Validation: Actual vs Predicted Temperature with h = {h}", "Temperature (°C)", baseline_T)

    print_error_table(horizon, T_model_errors, T_baseline_errors, "Temperature (°C)")

    # ===================================================
    # SECTION 7.3 — WIND SPEED BASELINE MODEL 
    # ===================================================
    print_header("SECTION 7.4 — WIND SPEED BASELINE MODEL")
    
    W_baseline_errors = []
    for h in horizon:
        df_h = df.copy()
        df_h["target_W"] = df_h["W"].shift(-h)
        df_h = df_h.dropna()
        train, val = split_train_val(df_h, val_hours)
        y_pred_W, y_val_W = build_prediction_model(train, val, features, "target_W")
        baseline_W = val["W"].to_numpy()
        W_baseline_errors.append(compute_errors(y_val_W, baseline_W))
        plot_actual_vs_predicted(y_val_W, y_pred_W, f"Validation: Actual vs Predicted Wind Speed with h = {h}", "Wind Speed (km/h)", baseline_W)

    print_error_table(horizon, W_model_errors, W_baseline_errors, "Wind Speed (km/h)")

    # ===================================================
    # SECTION 8 — FEATURE SELECTION 
    # ===================================================
    print_header("SECTION 8 — FEATURE SELECTION")

    # intiate current error
    y_pred_T_testing, y_val_T_testing = build_prediction_model(train, val, features, "target_T")
    y_pred_W_testing, y_val_W_testing = build_prediction_model(train, val, features, "target_W")

    # temperature
    rmse_T_testing = np.sqrt(np.mean((y_val_T_testing - y_pred_T_testing)**2))
    mae_T_testing  = np.mean(np.abs(y_val_T_testing - y_pred_T_testing))

    # wind
    rmse_W_testing = np.sqrt(np.mean((y_val_W_testing - y_pred_W_testing)**2))
    mae_W_testing  = np.mean(np.abs(y_val_W_testing - y_pred_W_testing))

    # for plotting how error decreases - initiate list
    total_error_sum =[rmse_T_testing + mae_T_testing + rmse_W_testing + mae_W_testing,]
    
    for i in range(len(additional_features)):
        # add additional features one by one to analyze their effect on predictions
        trial_features = features + [additional_features[i]]
        
        # temperature
        y_pred_T_new_feature, y_val_T_new_feature = build_prediction_model(train, val, trial_features, "target_T")

        rmse_T_new_feature = np.sqrt(np.mean((y_val_T_new_feature - y_pred_T_new_feature)**2))  # RMSE error
        mae_T_new_feature = np.mean(np.abs(y_val_T_new_feature - y_pred_T_new_feature))  # MAE error

        # wind
        y_pred_W_new_feature, y_val_W_new_feature = build_prediction_model(train, val, trial_features, "target_W")

        rmse_W_new_feature = np.sqrt(np.mean((y_val_W_new_feature - y_pred_W_new_feature)**2))  # RMSE error
        mae_W_new_feature = np.mean(np.abs(y_val_W_new_feature - y_pred_W_new_feature))  # MAE error

        # if error decreases overall with the new feature addition, add it to X
        error_sum = rmse_T_new_feature + mae_T_new_feature + rmse_W_new_feature + mae_W_new_feature

        total_error_sum.append(error_sum)

        if error_sum < (rmse_T_testing + mae_T_testing + rmse_W_testing + mae_W_testing):
            # features.append(additional_features[i])
            features = trial_features

            # update current error
            rmse_T_testing = rmse_T_new_feature
            mae_T_testing = mae_T_new_feature
            rmse_W_testing = rmse_W_new_feature
            mae_W_testing = mae_W_new_feature

    # accepted features
    print(f"Accepted features for horizon = {horizon}: {features}")

    print("RMSE Temperature (optimized features):", rmse_T_new_feature)
    print("MAE Temperature (optimized features):", mae_T_new_feature)
    print("RMSE Wind Speed (optimized features):", rmse_W_new_feature)
    print("MAE Wind Speed (optimized features):", mae_W_new_feature)

    # plotting error improvement
    x_labels = ["Original"] + additional_features

    plt.plot(x_labels, total_error_sum)

    plt.xlabel("Additional Feature Added")
    plt.ylabel("Total Error Sum")
    plt.title("How Total Error Changes with Additional Features")

    for i, value in enumerate(total_error_sum):
        plt.text(i, value + 0.01, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

    plt.show()

    # ===================================================
    # SECTION 9 — SEASONAL ANALYSIS
    # ===================================================
    print_header("SECTION 8 — FEATURE SELECTION")

    # checking the variability of temperature in the winter vs. the summer
    winter_T_std = np.std(fetch_open_meteo_hourly("2025-01-01", "2025-02-28")["temperature_2m"])
    summer_T_std = np.std(fetch_open_meteo_hourly("2025-07-01", "2025-08-31")["temperature_2m"])
    print(f"Standard deviation of temperature in the winter (Jan 1-Feb 28): {winter_T_std}")
    print(f"Standard deviation of temperature in the summer (Jul 1-Aug 31): {summer_T_std}")

    # checking variabiltiy of wind speed in the winter vs. summer
    winter_W_std = np.std(fetch_open_meteo_hourly("2025-01-01", "2025-02-28")["wind_speed_10m"])
    summer_W_std = np.std(fetch_open_meteo_hourly("2025-07-01", "2025-08-31")["wind_speed_10m"])
    print(f"Standard deviation of wind speed in the winter (Jan 1-Feb 28): {winter_W_std}")
    print(f"Standard deviation of wind speed in the summer (Jul 1-Aug 31): {summer_W_std}")
