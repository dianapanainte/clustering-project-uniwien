import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import sys


TRAIN_FILE   = sys.argv[1]
OUTPUT_FILE  = sys.argv[2]
FORECAST_YEAR = 2024

LAG_DAYS     = [1, 2, 3, 7, 14, 30]
ROLLING_WINS = [7, 14, 30]

LGBM_PARAMS = {
    'objective':        'regression',
    'metric':           'huber',
    'n_estimators':     500,
    'learning_rate':    0.05,
    'num_leaves':       63,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'n_jobs':           -1,
    'verbose':          -1,
}


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(id_vars='ID', var_name='ds', value_name='y')
    long['ds'] = pd.to_datetime(long['ds'], format='%Y-%m-%d')
    return long.sort_values(['ID', 'ds']).reset_index(drop=True)


def make_features(long: pd.DataFrame) -> pd.DataFrame:
    df = long.copy()

    df['dayofweek']  = df['ds'].dt.dayofweek
    df['month']      = df['ds'].dt.month
    df['dayofyear']  = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['quarter']    = df['ds'].dt.quarter

    for lag in LAG_DAYS:
        df[f'lag_{lag}'] = df.groupby('ID')['y'].shift(lag)

    for win in ROLLING_WINS:
        shifted = df.groupby('ID')['y'].shift(1)
        df[f'roll_mean_{win}'] = shifted.groupby(df['ID']).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        df[f'roll_std_{win}'] = shifted.groupby(df['ID']).transform(
            lambda x: x.rolling(win, min_periods=1).std()
        )

    return df


def preprocess(df: pd.DataFrame, zero_threshold: float = 0.8):
    consumption = df.drop(columns='ID')
    zero_frac   = (consumption == 0).sum(axis=1) / consumption.shape[1]

    all_zero_mask = zero_frac == 1.0
    sparse_mask   = (zero_frac >= zero_threshold) & ~all_zero_mask

    removed_ids = df.loc[all_zero_mask, 'ID'].tolist()
    flagged_ids = df.loc[sparse_mask,   'ID'].tolist()

    if removed_ids:
        print(f"  Removed {len(removed_ids)} all-zero households.")
    if flagged_ids:
        print(f"  Flagged {len(flagged_ids)} sparse households (>{zero_threshold*100:.0f}% zeros).")

    cleaned = df[~all_zero_mask].copy()

    id_col   = cleaned['ID'].reset_index(drop=True)
    cons_col = cleaned.drop(columns='ID').replace(0, np.nan).reset_index(drop=True)
    cleaned  = pd.concat([id_col, cons_col], axis=1)

    return cleaned, flagged_ids


def long_to_wide(long: pd.DataFrame, value_col: str = 'y') -> pd.DataFrame:
    wide = long.pivot(index='ID', columns='ds', values=value_col)
    wide.columns = wide.columns.strftime('%Y-%m-%d')
    return wide.reset_index()


def calculate_mae(forecast_file: str, test_file: str) -> float:
    forecast = pd.read_csv(forecast_file)
    test     = pd.read_csv(test_file)

    f_long = forecast.melt(id_vars='ID', var_name='Date', value_name='Prediction')
    t_long = test.melt(id_vars='ID', var_name='Date', value_name='Actual')

    merged = pd.merge(f_long, t_long, on=['ID', 'Date'])
    return mean_absolute_error(merged['Actual'], merged['Prediction'])


def feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {'ID', 'ds', 'y'}
    return [c for c in df.columns if c not in exclude]


def run_lgbm_forecast(
    train_file:    str = TRAIN_FILE,
    output_file:   str = OUTPUT_FILE,
    forecast_year: int = FORECAST_YEAR,
    lgbm_params:   dict = LGBM_PARAMS,
) -> pd.DataFrame:
    print("Loading training data...")
    raw_train = pd.read_csv(train_file, sep=',')
    train_df, flagged = preprocess(raw_train)
    print(f"  {len(train_df)} households retained for training.")

    print("Engineering features...")
    long_train  = wide_to_long(train_df)
    feat_train  = make_features(long_train).dropna()

    FEAT_COLS = feature_cols(feat_train)
    X_train   = feat_train[FEAT_COLS]
    y_train   = feat_train['y']

    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)
    print("  Training complete.")
    print(f"Generating rolling forecast for {forecast_year}...")

    history = long_train.copy()

    forecast_dates = pd.date_range(
        start=f'{forecast_year}-01-01',
        end=f'{forecast_year}-12-31',
        freq='D'
    )

    all_predictions = []

    for date in forecast_dates:
        new_rows = pd.DataFrame({
            'ID': train_df['ID'].values,
            'ds': date,
            'y':  np.nan
        })

        combined    = pd.concat([history, new_rows], ignore_index=True)
        feat_all    = make_features(combined)

        today_feats = feat_all[feat_all['ds'] == date][FEAT_COLS]

        preds = model.predict(today_feats)

        new_rows['y'] = preds
        all_predictions.append(new_rows.copy())

        history = pd.concat([history, new_rows], ignore_index=True)

    print("Saving forecast...")
    forecast_long = pd.concat(all_predictions, ignore_index=True)
    forecast_wide = long_to_wide(forecast_long, value_col='y')

    forecast_wide.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")

    return forecast_wide


forecast_df = run_lgbm_forecast()
