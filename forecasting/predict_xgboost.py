import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import sys
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DEFAULT_XGB_PARAMS = {
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
}


def _get_lag(consumption: np.ndarray, dates: pd.DatetimeIndex,
             target_date: pd.Timestamp, days_back: int,
             fallback: np.ndarray) -> np.ndarray:
    lag_date = target_date - pd.Timedelta(days=days_back)
    idx = np.where(dates == lag_date)[0]
    return consumption[:, idx[0]] if len(idx) else fallback


def _build_dataset(consumption: np.ndarray, dates: pd.DatetimeIndex,
                   target_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    n_hh            = consumption.shape[0]
    household_means = consumption.mean(axis=1)
    household_stds  = consumption.std(axis=1)
    eps             = 1e-6

    all_X, all_y = [], []

    for h, tdate in enumerate(target_dates):
        if h % 60 == 0:
            logger.info(f"  Building features for horizon {h}/{len(target_dates)} ({tdate.date()})")

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
                            sin_dow,  cos_dow], dtype=np.float32)
        cal_mat = np.tile(cal_row, (n_hh, 1))

        lag1   = _get_lag(consumption, dates, tdate,   1, household_means)
        lag7   = _get_lag(consumption, dates, tdate,   7, household_means)
        lag14  = _get_lag(consumption, dates, tdate,  14, household_means)
        lag30  = _get_lag(consumption, dates, tdate,  30, household_means)

        days_back_364 = 363 if (tdate.month == 2 and tdate.day == 29) else 364
        lag364 = _get_lag(consumption, dates, tdate, days_back_364, household_means)

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
            lag1,   lag7,   lag14,   lag30,   lag364,
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


class HouseholdForecaster:
    def __init__(
        self,
        id_column:  str  = "ID",
        start_date: str  = "2023-01-01",
        val_frac:   float = 0.1,
        xgb_params: dict | None = None,
        seed:       int  = 42,
    ):
        self.id_column  = id_column
        self.start_date = start_date
        self.val_frac   = val_frac
        self.seed       = seed
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.xgb_params["random_state"] = seed

        self.model_:       xgb.XGBRegressor | None = None
        self.consumption_: np.ndarray | None        = None
        self.ids_:         list | None              = None
        self.dates_:       pd.DatetimeIndex | None  = None

    def _load(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        self.ids_         = df[self.id_column].tolist()
        self.consumption_ = df.drop(columns=[self.id_column]).values.astype(np.float32)
        n_days            = self.consumption_.shape[1]
        self.dates_       = pd.date_range(self.start_date, periods=n_days, freq="D")
        logger.info(
            f"Loaded {len(self.ids_)} households | {n_days} days "
            f"({self.dates_[0].date()} → {self.dates_[-1].date()})"
        )

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info(f"\nTraining XGBoost on {X.shape[0]:,} samples × {X.shape[1]} features")

        n_val   = max(1, int(X.shape[0] * self.val_frac))
        n_train = X.shape[0] - n_val

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train:], y[n_train:]

        self.model_ = xgb.XGBRegressor(**self.xgb_params)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        val_pred = self.model_.predict(X_val)
        val_mae  = np.mean(np.abs(y_val - val_pred))
        logger.info(f"Validation MAE: {val_mae:.4f} kWh")

    def fit(self, csv_path: str, target_dates: pd.DatetimeIndex | None = None) -> "HouseholdForecaster":
        self._load(csv_path)

        if target_dates is None:
            target_dates = pd.date_range("2024-01-01", periods=366, freq="D")

        logger.info(f"\nBuilding feature matrix for {len(target_dates)} horizons...")
        X, y = _build_dataset(self.consumption_, self.dates_, target_dates)
        logger.info(f"Feature matrix: {X.shape}  |  Target vector: {y.shape}")

        self._train(X, y)
        self._last_target_dates = target_dates
        self._last_X            = X
        return self

    def predict(
        self,
        target_dates: pd.DatetimeIndex | None = None,
        csv_path: str | None = None,
    ) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        if target_dates is None:
            target_dates = self._last_target_dates
            X = self._last_X
        else:
            logger.info(f"\nBuilding feature matrix for new target dates ({len(target_dates)} horizons)...")
            X, _ = _build_dataset(self.consumption_, self.dates_, target_dates)

        logger.info("\nGenerating predictions...")
        n_hh      = len(self.ids_)
        n_horizon = len(target_dates)

        preds = self.model_.predict(X)
        preds = np.clip(preds, 0, None).reshape(n_horizon, n_hh).T

        date_cols = target_dates.strftime("%Y-%m-%d").tolist()
        out_df    = pd.DataFrame(preds, columns=date_cols)
        out_df.insert(0, "ID", self.ids_)

        if csv_path:
            out_df.to_csv(csv_path, index=False)
            logger.info(f"Saved → {csv_path}  shape={out_df.shape}")

        return out_df


def _run_cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler("xgb_forecast.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if len(sys.argv) != 3:
        print("Usage: python xgb_forecast.py <input.csv> <output.csv>")
        sys.exit(1)

    csv_input  = sys.argv[1]
    csv_output = sys.argv[2]

    logger.info("=" * 60)
    logger.info(f"Input:  {csv_input}")
    logger.info(f"Output: {csv_output}")
    logger.info("=" * 60)

    forecaster = HouseholdForecaster()
    forecaster.fit(csv_input)
    out_df = forecaster.predict(csv_path=csv_output)

    logger.info("\nSample (first 3 households, first 7 days):")
    logger.info(out_df.iloc[:3, :8].to_string(index=False))


if __name__ == "__main__":
    _run_cli()
