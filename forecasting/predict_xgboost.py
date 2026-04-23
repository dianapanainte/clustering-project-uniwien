import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import sys
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

CONFIG = {
    "csv_path":   sys.argv[1],
    "id_column":  "ID",
    "start_date": "2023-01-01",
    "csv_output": sys.argv[2],
    "seed":       42,
    "xgb_params": {
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "tree_method":      "hist",
        "device":           "cuda",
    },
}


def load_data(cfg: dict):
    df = pd.read_csv(cfg["csv_path"])
    ids         = df[cfg["id_column"]].tolist()
    consumption = df.drop(columns=[cfg["id_column"]]).values.astype(np.float32)
    n_days      = consumption.shape[1]
    dates       = pd.date_range(cfg["start_date"], periods=n_days, freq="D")
    logger.info(f"Loaded {len(ids)} households | {n_days} days "
                f"({dates[0].date()} → {dates[-1].date()})")
    return consumption, ids, dates


def get_lag(consumption: np.ndarray, dates: pd.DatetimeIndex,
            target_date: pd.Timestamp, days_back: int,
            fallback: np.ndarray) -> np.ndarray:
    lag_date = target_date - pd.Timedelta(days=days_back)
    idx = np.where(dates == lag_date)[0]
    return consumption[:, idx[0]] if len(idx) else fallback


def build_dataset(consumption: np.ndarray, dates: pd.DatetimeIndex,
                  target_dates: pd.DatetimeIndex) -> tuple:
    n_hh             = consumption.shape[0]
    household_means  = consumption.mean(axis=1)
    household_stds   = consumption.std(axis=1)
    eps              = 1e-6

    all_X, all_y = [], []

    for h, tdate in enumerate(target_dates):
        if h % 60 == 0:
            logger.info(f"  Building features for horizon {h}/365 ({tdate.date()})")

        dow        = tdate.dayofweek
        doy        = tdate.day_of_year
        month      = tdate.month
        is_weekend = int(dow >= 5)

        sin_doy1 = np.sin(2 * np.pi * 1 * doy / 365.25)
        cos_doy1 = np.cos(2 * np.pi * 1 * doy / 365.25)
        sin_doy2 = np.sin(2 * np.pi * 2 * doy / 365.25)
        cos_doy2 = np.cos(2 * np.pi * 2 * doy / 365.25)
        sin_dow  = np.sin(2 * np.pi * dow / 7)
        cos_dow  = np.cos(2 * np.pi * dow / 7)

        cal_row = np.array([dow, doy, month, is_weekend,
                            sin_doy1, cos_doy1, sin_doy2, cos_doy2,
                            sin_dow, cos_dow], dtype=np.float32)
        cal_mat = np.tile(cal_row, (n_hh, 1))

        lag1   = get_lag(consumption, dates, tdate,   1, household_means)
        lag7   = get_lag(consumption, dates, tdate,   7, household_means)
        lag14  = get_lag(consumption, dates, tdate,  14, household_means)
        lag30  = get_lag(consumption, dates, tdate,  30, household_means)

        if tdate.month == 2 and tdate.day == 29:
            lag364 = get_lag(consumption, dates, tdate, 363, household_means)
        else:
            lag364 = get_lag(consumption, dates, tdate, 364, household_means)

        lag1_n   = lag1   / (household_means + eps)
        lag7_n   = lag7   / (household_means + eps)
        lag364_n = lag364 / (household_means + eps)

        dow_mask   = dates.dayofweek == dow
        month_mask = dates.month == month

        dow_mean   = (consumption[:, dow_mask].mean(axis=1)
                      if dow_mask.sum() > 0 else household_means)
        month_mean = (consumption[:, month_mask].mean(axis=1)
                      if month_mask.sum() > 0 else household_means)

        hh_mat = np.column_stack([
            household_means, household_stds,
            lag1,   lag7,  lag14, lag30, lag364,
            lag1_n, lag7_n, lag364_n,
            dow_mean, month_mean,
        ]).astype(np.float32)

        X_h = np.hstack([cal_mat, hh_mat])

        if tdate.month == 2 and tdate.day == 29:
            idx28 = np.where(dates == pd.Timestamp("2023-02-28"))[0]
            idx01 = np.where(dates == pd.Timestamp("2023-03-01"))[0]
            if len(idx28) and len(idx01):
                y_h = (consumption[:, idx28[0]] + consumption[:, idx01[0]]) / 2
            elif len(idx28):
                y_h = consumption[:, idx28[0]]
            else:
                y_h = household_means
        else:
            same_day_2023 = tdate.replace(year=2023)
            idx = np.where(dates == same_day_2023)[0]
            y_h = consumption[:, idx[0]] if len(idx) else household_means

        all_X.append(X_h)
        all_y.append(y_h.astype(np.float32))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y


def train(X: np.ndarray, y: np.ndarray, cfg: dict) -> xgb.XGBRegressor:
    logger.info(f"\nTraining XGBoost on {X.shape[0]:,} samples × {X.shape[1]} features")

    n_total  = X.shape[0]
    n_val    = max(1, int(n_total * 0.1))
    n_train  = n_total - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:], y[n_train:]

    model = xgb.XGBRegressor(**cfg["xgb_params"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_pred = model.predict(X_val)
    val_mae  = np.mean(np.abs(y_val - val_pred))
    logger.info(f"Validation MAE: {val_mae:.4f} kWh")
    return model


def predict(model: xgb.XGBRegressor, X: np.ndarray,
            ids: list, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    n_hh      = len(ids)
    n_horizon = len(target_dates)

    preds = model.predict(X)
    preds = np.clip(preds, 0, None)
    preds = preds.reshape(n_horizon, n_hh).T

    date_cols = target_dates.strftime("%Y-%m-%d").tolist()
    out_df    = pd.DataFrame(preds, columns=date_cols)
    out_df.insert(0, "ID", ids)
    return out_df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler("xgb_forecast.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    cfg = CONFIG
    logger.info("=" * 60)
    logger.info(f"Input:  {cfg['csv_path']}")
    logger.info(f"Output: {cfg['csv_output']}")
    logger.info("=" * 60)

    consumption, ids, dates = load_data(cfg)

    dates_2024 = pd.date_range("2024-01-01", periods=366, freq="D")
    logger.info(f"\nBuilding feature matrix for {len(dates_2024)} horizons...")
    X, y = build_dataset(consumption, dates, dates_2024)
    logger.info(f"Feature matrix: {X.shape}  |  Target vector: {y.shape}")

    model = train(X, y, cfg)

    logger.info("\nGenerating 2024 predictions...")
    out_df = predict(model, X, ids, dates_2024)

    out_df.to_csv(cfg["csv_output"], index=False)
    logger.info(f"\nSaved → {cfg['csv_output']}  shape={out_df.shape}")
    logger.info("\nSample (first 3 households, first 7 days):")
    logger.info(out_df.iloc[:3, :8].to_string(index=False))


if __name__ == "__main__":
    main()
    