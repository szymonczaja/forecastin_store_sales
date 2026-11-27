import pandas as pd
import numpy as np
from .external_data import load_external_data, merge_external_data

def calendar_features(X: pd.DataFrame) -> pd.DataFrame:
    """Creating and adding calendar features to dataframe."""
    X = X.copy()
    date = X["date"]
    X["year"] = date.dt.year
    X["month"] = date.dt.month
    X["day_of_week"] = date.dt.dayofweek + 1
    X["day_of_month"] = date.dt.day
    X["day_of_year"] = date.dt.dayofyear
    X["week_of_year"] = date.dt.isocalendar().week.astype("int16")
    X["quarter"] = date.dt.quarter
    X["is_weekend"] = X["day_of_week"].isin([6, 7]).astype("int8")
    X["is_month_start"] = date.dt.is_month_start.astype("int8")
    X["is_month_end"] = date.dt.is_month_end.astype("int8")
    X["is_quarter_start"] = date.dt.is_quarter_start.astype("int8")
    X["is_quarter_end"] = date.dt.is_quarter_end.astype("int8")
    X["is_payday"] = ((date.dt.day == 15) | date.dt.is_month_end).astype("int8")
    X["days_to_month_end"] = date.dt.days_in_month - date.dt.day
    X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    return X


def add_lag_features(X: pd.DataFrame, group_cols: list, lags: list) -> pd.DataFrame:
    """Adding lag features to dataframe."""
    X = X.copy()
    grouped = X.groupby(group_cols, observed=True)["sales"]
    for lag in lags:
        X[f"lag_{lag}"] = grouped.shift(lag)
    return X


def add_rolling_features(X: pd.DataFrame, group_cols: list, horizon: int, rolling_windows: list, rolling_std_window: int) -> pd.DataFrame:
    """Adding leakage-safe rolling sales statistics."""
    X = X.copy()
    shifted = X.groupby(group_cols, observed=True)["sales"].shift(horizon)
    grouped = shifted.groupby([X[col] for col in group_cols], observed=True)
    for window in rolling_windows:
        X[f"rolling_mean_{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())
    X[f"rolling_std_{rolling_std_window}"] = grouped.transform(lambda s: s.rolling(rolling_std_window, min_periods=rolling_std_window).std())
    X["rolling_median_7"] = grouped.transform(lambda s: s.rolling(7, min_periods=7).median())
    X["rolling_median_28"] = grouped.transform(lambda s: s.rolling(28, min_periods=28).median())
    X["rolling_min_28"] = grouped.transform(lambda s: s.rolling(28, min_periods=28).min())
    X["rolling_max_28"] = grouped.transform(lambda s: s.rolling(28, min_periods=28).max())
    X["zero_sales_ratio_28"] = grouped.transform(lambda s: (s == 0).rolling(28, min_periods=28).mean())
    mean_7 = grouped.transform(lambda s: s.rolling(7, min_periods=7).mean())
    mean_28 = grouped.transform(lambda s: s.rolling(28, min_periods=28).mean())
    mean_84 = grouped.transform(lambda s: s.rolling(84, min_periods=84).mean())
    X["sales_trend_7_28"] = mean_7 / (mean_28 + 1e-6)
    X["sales_trend_28_84"] = mean_28 / (mean_84 + 1e-6)
    return X


def add_promotion_features(X: pd.DataFrame, group_cols: list, horizon: int) -> pd.DataFrame:
    """Adding promotion-related features."""
    X = X.copy()
    X["is_promotion"] = (X["onpromotion"].fillna(0) > 0).astype("int8")
    shifted = X.groupby(group_cols, observed=True)["onpromotion"].shift(horizon)
    grouped = shifted.groupby([X[col] for col in group_cols], observed=True)
    X[f"promotion_lag_{horizon}"] = shifted
    X["promotion_rolling_mean_28"] = grouped.transform(lambda s: s.rolling(28, min_periods=28).mean())
    return X


def sort_data(X: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Sorting dataframe before feature engineering."""
    return X.sort_values(by=group_cols + ["date"]).copy()


def preprocessor(X: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Prepare data and generate features."""
    group_cols = feature_config["group_columns"]
    X_sorted = sort_data(X, group_cols)
    X_sorted = calendar_features(X_sorted)
    X_sorted = add_lag_features(X_sorted, group_cols, feature_config["lags"])
    X_sorted = add_rolling_features(X_sorted, group_cols, feature_config["horizon"], feature_config["rolling_windows"], feature_config["rolling_std_window"])
    X_sorted = add_promotion_features(X_sorted, group_cols, feature_config["horizon"])
    return X_sorted


def convert_object_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Converting object features to category because of LGBM."""
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].astype("category")
    return df


def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Combining training and test data."""
    train = train.assign(is_test=0)
    test = test.assign(is_test=1)
    return pd.concat([train, test], ignore_index=True)


def split_data(df: pd.DataFrame, val_start) -> tuple:
    """Splitting data to train, validation and test datasets."""
    val_start = pd.Timestamp(val_start)
    known = df[df["is_test"] == 0]
    train = known[known["date"] < val_start]
    val = known[known["date"] >= val_start]
    test = df[df["is_test"] == 1]
    drop_cols = ["id", "date", "sales", "is_test"]
    return (train.drop(columns=drop_cols), train["sales"], val.drop(columns=drop_cols), val["sales"], test.drop(columns=drop_cols), test["id"])


def load_raw_data(train_path: str, test_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loading raw data from path."""
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    if test_path:
        test_df = pd.read_csv(test_path, parse_dates=["date"])
        return train_df, test_df    
    return train_df


def prepare_and_split_data(train, test, config):
    val_start = config["data"]["validation_start"]
    feature_config = config["features"]
    df = combine_train_test(train, test)
    df = merge_external_data(df, load_external_data(config["data"]["data_dir"]), lags=feature_config['lags'], rolling_windows=feature_config['rolling_windows'], horizon=feature_config['horizon'])
    df_preprocessed = preprocessor(df, feature_config)
    df_preprocessed = convert_object_to_category(df_preprocessed)
    return split_data(df_preprocessed, val_start)