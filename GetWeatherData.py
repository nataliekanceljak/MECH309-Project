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

if __name__ == "__main__":
    # use data caputring all seasons 
    start_date = "2024-06-01"
    end_date = "2025-08-31"
    
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

    horizon = 3    # number of hours to predict into future
    
    # Features vector
    features = [
    "T", "W",
    "T_lag1", "T_lag2", "T_lag3", "T_lag6", "T_lag12", "T_lag24",
    "W_lag1", "W_lag3", "W_lag6", "W_lag12",
    "sin_day", "cos_day",
    "sin_year", "cos_year",
    ]

    # Temperature model
    df["target_T"] = df["T"].shift(-horizon)

    df_model = df.dropna()  # remove data with missing values

    val_hours = 92 * 24  # train Jun 1 2024-Apr 30 2025, validate Jun 1 2025-Aug 31 2025
    train, val = split_train_val(df_model, val_hours)
    
    y_pred_T, y_val_T = build_prediction_model(train, val, features, "target_T")

    rmse_T = np.sqrt(np.mean((y_val_T - y_pred_T)**2))  # RMSE error
    mae_T = np.mean(np.abs(y_val_T - y_pred_T))  # MAE error
    print(f"RMSE Temperature:", rmse_T)
    print(f"MAE Temperature:", mae_T)

    # Plot of TEMP data for chosen interval
    plt.figure()
    df["T"].plot(linewidth=1)
    plt.title("Montreal hourly temperature (2m)")
    plt.ylabel("T (°C)")
    plt.tight_layout()
    plt.show()

    # Plot of true vs. predicted TEMP val data
    plt.figure()
    plt.plot(y_val_T, label="Actual")
    plt.plot(y_pred_T, label="Predicted")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.title("Validation: Actual vs Predicted Temperature")
    plt.legend()
    plt.show()

    # Wind speed model
    df["target_W"] = df["W"].shift(-horizon)
    df_model = df.dropna()

    train, val = split_train_val(df_model, val_hours)
    
    y_pred_W, y_val_W = build_prediction_model(train, val, features, "target_W")

    rmse_W = np.sqrt(np.mean((y_val_W - y_pred_W)**2))  # RMSE error
    mae_W = np.mean(np.abs(y_val_W - y_pred_W))  # MAE error
    print(f"RMSE Wind Speed:", rmse_W)
    print(f"MAE Wind Speed:", mae_W)

    # Plot of WIND data for chosen interval
    plt.figure()
    df["W"].plot(linewidth=1)
    plt.title("Montreal hourly wind speed (10m)")
    plt.ylabel("W (km/h)")
    plt.tight_layout()
    plt.show()

    # Plot of true vs. predicted WIND val data
    plt.figure()
    plt.plot(y_val_W, label="Actual")
    plt.plot(y_pred_W, label="Predicted")
    plt.xlabel("Time (hours)")
    plt.ylabel("Wind speed (km/h))")
    plt.title("Validation: Actual vs Predicted Wind Speed")
    plt.legend()
    plt.show()

    # Baseline model for temperature
    baseline_T = val["T"].to_numpy()

    plt.figure()

    plt.plot(y_val_T, label="Actual")
    plt.plot(y_pred_T, label="Model")
    plt.plot(baseline_T, label="Baseline")

    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.title("Actual vs Model vs Baseline for Temperature")

    plt.legend()
    plt.show()

    rmse_base_T = np.sqrt(np.mean((y_val_T - baseline_T)**2))
    mae_base_T  = np.mean(np.abs(y_val_T - baseline_T))

    print(f"Baseline RMSE Temperature:", rmse_base_T)
    print(f"Baseline MAE Temperature:", mae_base_T)

    # Baseline model for wind
    baseline_W = val["W"].to_numpy()

    plt.figure()

    plt.plot(y_val_W, label="Actual")
    plt.plot(y_pred_W, label="Model")
    plt.plot(baseline_W, label="Baseline")

    plt.xlabel("Time (hours)")
    plt.ylabel("Wind speed (km/h)")
    plt.title("Actual vs Model vs Baseline for Wind Speed")

    plt.legend()
    plt.show()
    
    rmse_base_W = np.sqrt(np.mean((y_val_W - baseline_W)**2))
    mae_base_W  = np.mean(np.abs(y_val_W - baseline_W))

    print(f"Baseline RMSE Wind Speed:", rmse_base_W)
    print(f"Baseline MAE Wind Speed:", mae_base_W)
    
    # testing which features improve the prediction
    additional_features = ["Wd", "RH", "P", "Prec", "Cloud"]
    features_testing = [
    "T", "W",
    "T_lag1", "T_lag2", "T_lag3", "T_lag6", "T_lag12", "T_lag24",
    "W_lag1", "W_lag3", "W_lag6", "W_lag12",
    "sin_day", "cos_day",
    "sin_year", "cos_year",
    ]

    # intiate current error
    y_pred_T_testing, y_val_T_testing = build_prediction_model(train, val, features_testing, "target_T")
    y_pred_W_testing, y_val_W_testing = build_prediction_model(train, val, features_testing, "target_W")

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
        trial_features = features_testing + [additional_features[i]]
        
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
            features_testing = trial_features

            # update current error
            rmse_T_testing = rmse_T_new_feature
            mae_T_testing = mae_T_new_feature
            rmse_W_testing = rmse_W_new_feature
            mae_W_testing = mae_W_new_feature

    # accepted features
    print(f"Accepted features for horizon = {horizon}: {features_testing}")

    print("RMSE Temperature (updated features):", rmse_T_new_feature)
    print("MAE Temperature (updated features):", mae_T_new_feature)
    print("RMSE Wind Speed (updated features):", rmse_W_new_feature)
    print("MAE Wind Speed (updated features):", mae_W_new_feature)

    # plotting error improvement
    x_labels = ["original"] + additional_features

    plt.plot(x_labels, total_error_sum)

    plt.xlabel("Additional Feature Added")
    plt.ylabel("Total Error Sum")
    plt.title("How RMSE and MAE Change with Additional Features")

    plt.legend()
    plt.show()

