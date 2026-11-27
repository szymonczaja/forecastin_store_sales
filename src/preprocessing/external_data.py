import pandas as pd 
import numpy as np 
from pathlib import Path

def load_external_data(data_dir): 
    data_dir = Path(data_dir)
    stores = pd.read_csv(data_dir / "stores.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=['date'])
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=['date'])
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=['date'])
    stores = prepare_stores(stores)
    transactions = prepare_transactions(transactions)
    oil = prepare_oil(oil)
    holidays = prepare_holidays(holidays)
    return {
        "stores" : stores, 
        "transactions" : transactions, 
        "oil" : oil, 
        "holidays" : holidays, 
    }

def prepare_stores(stores: pd.DataFrame) -> pd.DataFrame: 
    stores_copy = stores.copy()
    if stores_copy["store_nbr"].duplicated().any():
        raise ValueError("Duplicate store_nbr values found!")
    if stores_copy.isna().any().any():
        raise ValueError("Missing values found in stores DataFrame!")
    stores_copy[["city", "state", "type"]] = stores_copy[["city", "state", "type"]].astype('category')
    return stores_copy

def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame: 
    transactions_copy = transactions.copy() 
    required_cols = {"date", "store_nbr", "transactions"}
    if not required_cols.issubset(transactions_copy.columns):
        raise ValueError("Missing cols in transactions DataFrame!")
    if transactions_copy.duplicated(subset=["date", "store_nbr"]).any():
        raise ValueError("Duplicate combination of store_nbr/date found!")
    if transactions_copy.isna().any().any():
        raise ValueError("Missing values found in transactions DataFrame!")
    if not pd.api.types.is_numeric_dtype(transactions_copy["transactions"]):
            raise ValueError("Transactions number must be numeric!")
    if (transactions_copy['transactions'] < 0).any(): 
        raise ValueError("Trasactions must be grater or equal to zero!")
    return transactions_copy

def prepare_oil(oil: pd.DataFrame) -> pd.DataFrame: 
    oil_copy = oil.copy() 
    required_cols = {'date', 'dcoilwtico'}
    if not required_cols.issubset(oil_copy.columns): 
        raise ValueError("Missing cols in oil DataFrame!")
    if oil_copy['date'].duplicated().any(): 
        raise ValueError("Duplicated date in oil DataFrame!")
    if not pd.api.types.is_datetime64_any_dtype(oil_copy['date']):
        raise ValueError("Date olumn must be datetime!")
    if not pd.api.types.is_numeric_dtype(oil_copy['dcoilwtico']):
        raise ValueError("Oil price must be numeric!")
    if (oil_copy['dcoilwtico'].dropna() < 0).any(): 
        raise ValueError("Oil price must be greater or equal to zero!")
    return oil_copy
    
def prepare_holidays(holidays: pd.DataFrame) -> pd.DataFrame: 
    holidays_copy = holidays.copy() 
    required_cols = {"date", "type", "locale", "locale_name", "description", "transferred"}
    if not required_cols.issubset(holidays_copy.columns):
        raise ValueError("Missing cols in holidays event DataFrame!")
    if holidays_copy.duplicated(subset=["date", "type", "locale", "locale_name", "description"]).any(): 
        raise ValueError("Duplicated records in holidays DataFrame!")
    if not pd.api.types.is_datetime64_any_dtype(holidays_copy['date']): 
        raise ValueError("Date must be datetime!")
    if holidays_copy[["date", "type", "locale", "locale_name", "description", "transferred"]].isna().any().any():
        raise ValueError("Missing values in holidays event!")
    return holidays_copy

def add_holidays_features(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    holidays_copy = holidays.copy()
    stores_copy = stores[["store_nbr", "city", "state"]].copy()
    holidays_copy = holidays_copy[holidays_copy["transferred"] == False]
    national = holidays_copy[holidays_copy["locale"] == "National"].copy()
    regional = holidays_copy[holidays_copy["locale"] == "Regional"].merge(
        stores_copy,
        left_on="locale_name",
        right_on="state",
        how="inner"
    )
    local = holidays_copy[holidays_copy["locale"] == "Local"].merge(
        stores_copy,
        left_on="locale_name",
        right_on="city",
        how="inner"
    )
    national = national.assign(
        is_national_holiday=1,
        is_regional_holiday=0,
        is_local_holiday=0
    )
    regional = regional.assign(
        is_national_holiday=0,
        is_regional_holiday=1,
        is_local_holiday=0
    )
    local = local.assign(
        is_national_holiday=0,
        is_regional_holiday=0,
        is_local_holiday=1
    )

    national = national.merge(stores_copy[["store_nbr"]], how="cross")
    holiday_features = pd.concat([national, regional, local], ignore_index=True)
    holiday_features = holiday_features.groupby(
        ["date", "store_nbr"],
        as_index=False
    ).agg(
        is_national_holiday=("is_national_holiday", "max"),
        is_regional_holiday=("is_regional_holiday", "max"),
        is_local_holiday=("is_local_holiday", "max")
    )
    holiday_features["is_holiday"] = (
        holiday_features[
            ["is_national_holiday", "is_regional_holiday", "is_local_holiday"]
        ].max(axis=1)
    )
    return holiday_features


def add_transactions_features(transactions: pd.DataFrame, lags: list, rolling_windows: list, horizon: int) -> pd.DataFrame:
    t_copy = transactions.sort_values(by=['store_nbr', 'date']).copy()
    for lag in lags:
        t_copy[f'transactions_lag_{lag}'] = t_copy.groupby('store_nbr')['transactions'].shift(lag)
    for window in rolling_windows:
        t_copy[f'transactions_roll_mean_{window}'] =  t_copy.groupby('store_nbr')['transactions'].transform(lambda x: x.shift(horizon).rolling(window).mean())
    return t_copy 

def add_oil_features(oil: pd.DataFrame, lags: list, rolling_windows: list, horizon: int) -> pd.DataFrame: 
    o_copy = oil.sort_values(by=['date']).copy() 
    for lag in lags: 
        o_copy[f"oil_lag_{lag}"] = o_copy['dcoilwtico'].shift(lag) 
    for window in rolling_windows: 
        o_copy[f"oil_roll_mean_{window}"] = o_copy['dcoilwtico'].shift(horizon).rolling(window).mean()
    return o_copy 

def merge_external_data(df: pd.DataFrame, external_data: dict, lags: list, rolling_windows: list, horizon: int) -> pd.DataFrame:
    df_copy = df.copy()
    transactions = add_transactions_features(external_data['transactions'], lags, rolling_windows, horizon)
    oil = add_oil_features(external_data['oil'], lags, rolling_windows, horizon)
    transactions = transactions.drop(columns=['transactions'])
    oil = oil.drop(columns=['dcoilwtico'])
    df_copy = df_copy.merge(external_data["stores"], on="store_nbr", how="left", validate="many_to_one")
    df_copy = df_copy.merge(transactions, on=["date", "store_nbr"], how="left", validate="many_to_one")
    df_copy = df_copy.merge(oil, on="date", how="left", validate="many_to_one")
    holidays = add_holidays_features(external_data["holidays"], external_data["stores"])
    df_copy = df_copy.merge(holidays, on=["date", "store_nbr"], how="left", validate="many_to_one")
    holiday_cols = [
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_holiday"]
    df_copy[holiday_cols] = df_copy[holiday_cols].fillna(0)
    return df_copy

