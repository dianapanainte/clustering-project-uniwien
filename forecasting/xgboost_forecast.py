import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "csv_path":    "sample_23.csv",
    "id_column":   "ID",
    "start_date":  "2023-01-01",
    "pred_year":   2024,
    "val_split":   0.1,
    "models_dir":  "xgb_models",
    "seed":        42,

    "xgb_params": {
        "n_estimators":      300,
        "max_depth":         6,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  3,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      42,
        "n_jobs":            -1,
        "tree_method":       "hist",
        "device":            "cuda",
    }
}

def make_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=dates)
    df["dayofyear"]  = dates.day_of_year
    df["dayofweek"]  = dates.dayofweek
    df["month"]      = dates.month
    df["quarter"]    = dates.quarter
    df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    df["is_month_start"] = dates.is_month_start.astype(int)
    df["is_month_end"]   = dates.is_month_end.astype(int)

    for k in [1, 2, 3]:
        df[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * dates.day_of_year / 365.25)
        df[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * dates.day_of_year / 365.25)

    return df.reset_index(drop=True)


def load_data(cfg: dict):
    df = pd.read_csv(cfg["csv_path"])
    ids         = df[cfg["id_column"]].tolist()
    consumption = df.drop(columns=[cfg["id_column"]]).values.astype(np.float32)

    dates = pd.date_range(cfg["start_date"], periods=consumption.shape[1], freq="D")
    return consumption, ids, dates

def build_training_data(consumption: np.ndarray, dates: pd.DatetimeIndex,
                        horizon: int, cal_features: pd.DataFrame):
    n_households = consumption.shape[0]

    hist_mean   = consumption.mean(axis=1)
    hist_std    = consumption.std(axis=1)
    hist_last   = consumption[:, -1]
    hist_last7  = consumption[:, -7:].mean(axis=1)
    hist_last30 = consumption[:, -30:].mean(axis=1)

    target_date   = pd.date_range("2024-01-01", periods=366, freq="D")[horizon]
    target_dow    = target_date.dayofweek
    same_dow_mask = dates.dayofweek == target_dow
    if same_dow_mask.sum() > 0:
        same_dow_mean = consumption[:, same_dow_mask].mean(axis=1)
    else:
        same_dow_mean = hist_mean

    target_month   = target_date.month
    same_month_mask = dates.month == target_month
    if same_month_mask.sum() > 0:
        same_month_mean = consumption[:, same_month_mask].mean(axis=1)
    else:
        same_month_mean = hist_mean

    cal_row = cal_features.iloc[horizon].values
    cal_mat = np.tile(cal_row, (n_households, 1))

    hh_features = np.column_stack([
        hist_mean,
        hist_std,
        hist_last,
        hist_last7,
        hist_last30,
        same_dow_mean,
        same_month_mean,
    ])

    X = np.hstack([cal_mat, hh_features])  # (H, n_cal + n_hh)
    y = consumption[:, horizon] if horizon < consumption.shape[1] else None

    return X.astype(np.float32), y

def train_all_horizons(consumption: np.ndarray, dates: pd.DatetimeIndex,
                       cal_2024: pd.DataFrame, cfg: dict):
    """Train one XGBoost model per forecast horizon and save to disk."""
    os.makedirs(cfg["models_dir"], exist_ok=True)

    n_households = consumption.shape[0]
    n_val        = max(1, int(n_households * cfg["val_split"]))
    n_train      = n_households - n_val

    val_rmse_list = []

    for h in range(366):
        X, y = build_training_data(consumption, dates, h, cal_2024)

        # We can only train on horizons within the 365-day history
        # For horizons >= 365, we use the last available day as proxy target
        if h < consumption.shape[1]:
            X_train, y_train = X[:n_train], y[:n_train]
            X_val,   y_val   = X[n_train:], y[n_train:]
        else:
            # Beyond available history: use last day as proxy (pipeline still works)
            proxy_y = consumption[:, -1]
            X_train, y_train = X[:n_train], proxy_y[:n_train]
            X_val,   y_val   = X[n_train:], proxy_y[n_train:]

        model = xgb.XGBRegressor(**cfg["xgb_params"])
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = model.predict(X_val)
        rmse     = mean_squared_error(y_val, val_pred) ** 0.5
        val_rmse_list.append(rmse)

        model_path = os.path.join(cfg["models_dir"], f"model_h{h:03d}.json")
        model.save_model(model_path)

        if h % 30 == 0 or h == 365:
            print(f"  Horizon {h:3d}/365 trained | Val RMSE: {rmse:.4f}")

    print(f"\nMean Val RMSE across all horizons: {np.mean(val_rmse_list):.4f}")
    return val_rmse_list

def predict_2024(consumption: np.ndarray, dates: pd.DatetimeIndex,
                 cal_2024: pd.DataFrame, ids: list, cfg: dict) -> pd.DataFrame:
    """Load each saved model and predict for all households."""
    all_preds = []

    for h in range(366):
        X, _ = build_training_data(consumption, dates, h, cal_2024)

        model_path = os.path.join(cfg["models_dir"], f"model_h{h:03d}.json")
        model = xgb.XGBRegressor(**cfg["xgb_params"])
        model.load_model(model_path)

        preds = model.predict(X)          # (n_households,)
        all_preds.append(preds)

    all_preds = np.array(all_preds).T

    dates_2024 = pd.date_range("2024-01-01", periods=366, freq="D").strftime("%Y-%m-%d").tolist()
    out_df = pd.DataFrame(all_preds, columns=dates_2024)
    out_df.insert(0, cfg["id_column"], ids)
    return out_df

def main():
    cfg = CONFIG

    print("Loading data...")
    consumption, ids, dates = load_data(cfg)
    print(f"  {len(ids)} households | {consumption.shape[1]} days of history")
    print(f"  History: {dates[0].date()} → {dates[-1].date()}")

    dates_2024 = pd.date_range("2024-01-01", periods=366, freq="D")
    cal_2024   = make_calendar_features(dates_2024)
    print(f"  Calendar features: {cal_2024.shape[1]} columns")

    print("\nTraining 366 XGBoost models...")
    val_rmse = train_all_horizons(consumption, dates, cal_2024, cfg)

    print("\nPredicting 2024...")
    out_df = predict_2024(consumption, dates, cal_2024, ids, cfg)

    out_df.to_csv("forecast_xgboost_2024.csv", index=False)
    print(f"Predictions saved to forecast_xgboost_2024.csv  {out_df.shape}")

    print("\nSample predictions (first 3 households, first 7 days of 2024):")
    print(out_df.iloc[:3, :8].to_string(index=False))


if __name__ == "__main__":
    main()
