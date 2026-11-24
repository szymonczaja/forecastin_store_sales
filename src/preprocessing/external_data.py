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
    if (transactions_copy['transactions'] < 0).any(): 
        raise ValueError("Trasactions must be grater or equal to zero!")
    if not pd.api.types.is_numeric_dtype(transactions_copy["transactions"]):
        raise ValueError("Transactions number must be numeric!")
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

def merge_external_data(df: pd.DataFrame, external_data: dict) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.merge(external_data["stores"], on="store_nbr", how="left", validate="many_to_one")
    df_copy = df_copy.merge(external_data["transactions"], on=["date", "store_nbr"], how="left", validate="many_to_one")
    df_copy = df_copy.merge(external_data["oil"], on="date", how="left", validate="many_to_one")
    return df_copy