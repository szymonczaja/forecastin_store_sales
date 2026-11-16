import pandas as pd 
import numpy as np


def calendar_features(X : pd.DataFrame) -> pd.DataFrame: 
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day_of_week'] = X['date'].dt.dayofweek + 1
    X['day_of_month'] = X['date'].dt.day
    X['is_weekend'] = np.where((X['day_of_week'] == 6) | (X['day_of_week'] == 7), 1, 0)
    return X 


def add_lag_features(X : pd.DataFrame, group_cols, lags) -> pd.DataFrame:
    grouped = X.groupby(group_cols)['sales']
    for lag in lags: 
        X[f'lag_{lag}'] = grouped.shift(lag) 
    return X 


def add_rolling_features(X : pd.DataFrame, group_cols, horizon, rolling_windows, rolling_std_window) -> pd.DataFrame: 
    shifted = X.groupby(group_cols)['sales'].shift(horizon) 
    grouped = shifted.groupby([X[c] for c in group_cols])
    for w in rolling_windows: 
        X[f'rolling_mean_{w}'] = grouped.transform(lambda s: s.rolling(w).mean()) 
    X[f'rolling_std_{rolling_std_window}'] = grouped.transform(lambda s: s.rolling(rolling_std_window).std())
    return X 


def sort_data(X: pd.DataFrame, group_cols) -> pd.DataFrame: 
    sort_cols = group_cols + ['date']
    return X.sort_values(by=sort_cols).copy()


def preprocessor(X : pd.DataFrame, feature_config) -> pd.DataFrame: 
    group_cols = feature_config['group_columns']
    lags = feature_config['lags']
    horizon = feature_config['horizon']
    rolling_windows = feature_config['rolling_windows']
    rolling_std_window = feature_config['rolling_std_window']
    X_sorted = sort_data(X, group_cols)
    X_sorted = calendar_features(X_sorted)
    X_sorted = add_lag_features(X_sorted, group_cols, lags)
    X_sorted = add_rolling_features(X_sorted, group_cols, horizon, rolling_windows, rolling_std_window)
    return X_sorted


def convert_object_to_category(df): 
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('category')
    return df 


def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame: 
    train = train.assign(is_test=0) 
    test = test.assign(is_test=1) 
    return pd.concat([train, test], ignore_index=True) 


def split_data(df: pd.DataFrame, val_start): 
    val_start = pd.Timestamp(val_start) 
    known = df[df['is_test'] == 0] 
    train = known[known['date'] < val_start] 
    val = known[known['date'] >= val_start]
    test = df[df['is_test'] == 1]
    drop_cols = ['id', 'date', 'sales', 'is_test']
    return (
        train.drop(columns=drop_cols), train['sales'], 
        val.drop(columns=drop_cols), val['sales'], 
        test.drop(columns=drop_cols), test['id']
    )


def load_raw_data(train_path, test_path) -> tuple[pd.DataFrame, pd.DataFrame]: 
    train_df = pd.read_csv(train_path, parse_dates=['date']) 
    test_df = pd.read_csv(test_path, parse_dates=['date']) 
    return (train_df, test_df) 
 

def prepare_and_split_data(train: pd.DataFrame, test: pd.DataFrame, config) -> tuple[pd.DataFrame, pd.Series]: 
    val_start = config['data']['validation_start']
    feature_config = config['features']
    df = combine_train_test(train, test) 
    df_preprocessed = preprocessor(df, feature_config) 
    df_preprocessed = convert_object_to_category(df_preprocessed) 
    return split_data(df_preprocessed, val_start)
