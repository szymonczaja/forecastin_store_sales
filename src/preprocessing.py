import pandas as pd 
import numpy as np 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER_PATH = PROJECT_ROOT / 'data' / 'raw'
HORIZON = 16
GROUP_COLS = ['store_nbr', 'family']
LAGS = [16, 21, 28, 364]
ROLLING_WINDOWS = [7, 14, 28]
VAL_START = pd.Timestamp('2017-07-31')
TRAIN_PATH = DATA_FOLDER_PATH / 'train.csv'
TEST_PATH = DATA_FOLDER_PATH / 'test.csv'

def calendar_features(X : pd.DataFrame) -> pd.DataFrame: 
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day_of_week'] = X['date'].dt.dayofweek + 1
    X['day_of_month'] = X['date'].dt.day
    X['is_weekend'] = np.where((X['day_of_week'] == 6) | (X['day_of_week'] == 7), 1, 0)
    return X 

def add_lag_features(X : pd.DataFrame) -> pd.DataFrame:
    grouped = X.groupby(GROUP_COLS)['sales']
    for lag in LAGS: 
        X[f'lag_{lag}'] = grouped.shift(lag) 
    return X 

def add_rolling_features(X : pd.DataFrame) -> pd.DataFrame: 
    shifted = X.groupby(GROUP_COLS)['sales'].shift(HORIZON) 
    grouped = shifted.groupby([X[c] for c in GROUP_COLS])
    for w in ROLLING_WINDOWS: 
        X[f'rolling_mean_{w}'] = grouped.transform(lambda s: s.rolling(w).mean()) 
    X['rolling_std_14'] = grouped.transform(lambda s: s.rolling(14).std())
    return X 

def sort_data(X: pd.DataFrame) -> pd.DataFrame: 
    return X.sort_values(by=['store_nbr', 'family', 'date']).copy()

def preprocessor(X : pd.DataFrame) -> pd.DataFrame: 
    X_sorted = sort_data(X)
    return add_rolling_features(add_lag_features(calendar_features(X_sorted)))


def convert_object_to_category(df): 
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('category')
    return df 


def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame: 
    train = train.assign(is_test=0) 
    test = test.assign(is_test=1) 
    return pd.concat([train, test], ignore_index=True) 

def split_data(df: pd.DataFrame): 
    known = df[df['is_test'] == 0] 
    train = known[known['date'] < VAL_START] 
    val = known[known['date'] >= VAL_START]
    test = df[df['is_test'] == 1]
    drop_cols = ['id', 'date', 'sales', 'is_test']
    return (
        train.drop(columns=drop_cols), train['sales'], 
        val.drop(columns=drop_cols), val['sales'], 
        test.drop(columns=drop_cols), test['id']
    )

def load_raw_data(train_path=TRAIN_PATH, test_path=TEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame]: 
    train_df = pd.read_csv(train_path, parse_dates=['date']) 
    test_df = pd.read_csv(test_path, parse_dates=['date']) 
    return (train_df, test_df) 
 

def prepare_and_split_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: 
    df = combine_train_test(train, test) 
    df_preprocessed = preprocessor(df) 
    df_preprocessed = convert_object_to_category(df_preprocessed) 
    return split_data(df_preprocessed)

