import pandas as pd 
import numpy as np
import pytest
from src.preprocessing import calendar_features, add_lag_features, add_rolling_features, convert_object_to_category, merge_additional_data

def test_calendar_features(): 
    df = pd.DataFrame({
        'date': pd.to_datetime(['2017-01-07'])
    })
    result = calendar_features(df) 
    assert result['year'].iloc[0] == 2017 
    assert result['month'].iloc[0] == 1 
    assert result['day_of_month'].iloc[0] == 7
    assert result['day_of_week'].iloc[0] == 6
    assert result['is_weekend'].iloc[0] == 1 

def test_add_lag_features(): 
    df = pd.DataFrame({
    'date': pd.date_range(start='2017-01-01', periods=15, freq='D'),
    'store_nbr': [1] * 15,
    'family': ['AUTOMOTIVE'] * 15,
    'sales': np.arange(10, 160, 10)
    })
    result = add_lag_features(df) 
    assert result['lag_7'].iloc[7] == 10 
    assert result['lag_7'].iloc[:7].isna().all()
    assert result['lag_14'].iloc[14] == 10 
    assert result['lag_7'].iloc[14] == 80 
    assert result['lag_14'].iloc[:14].isna().all()

def test_add_rolling_features(): 
    df = pd.DataFrame({
    'date': pd.date_range(start='2017-01-01', periods=15, freq='D'),
    'store_nbr': [1] * 15,
    'family': ['AUTOMOTIVE'] * 15,
    'sales': np.arange(10, 160, 10)})
    result = add_rolling_features(df) 
    assert np.isnan(result['rolling_mean_7'].iloc[6]) 
    assert result['rolling_mean_7'].iloc[7] == 40.0 
    assert result['rolling_mean_7'].iloc[8] == 50.0 
    assert np.isnan(result['rolling_mean_28'].iloc[14]) 
    assert np.isclose(result['rolling_std_7'].iloc[7], 21.6025, atol=0.001)

def test_convert_object_to_category(): 
    df = pd.DataFrame({
    'family': ['AUTOMOTIVE', 'GROCERY', 'BEVERAGES'],
    'city': ['Quito', 'Guayaquil', 'Quito'],
    'sales': [100, 200, 300]
    })
    result = convert_object_to_category(df) 
    assert result['family'].iloc[0] == 'AUTOMOTIVE'
    assert isinstance(result['family'].dtype, pd.CategoricalDtype)
    assert isinstance(result['city'].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_numeric_dtype(result['sales'])

def test_merge_additional_data(): 
    X = pd.DataFrame({
    'store_nbr': [1, 2, 3],
    'sales': [100, 200, 300]
    })
    data = pd.DataFrame({
    'store_nbr': [1, 2, 3],
    'city': ['Quito', 'Guayaquil', 'Cuenca'],
    'cluster': [13, 5, 8]
    })
    data_duplicate = pd.DataFrame({
    'store_nbr': [1, 1, 2],
    'city': ['Quito', 'Quito', 'Guayaquil'],
    'cluster': [13, 13, 5]
    })
    result = merge_additional_data(X, data, on=['store_nbr'])
    assert result.shape[0] == 3
    assert 'city' in result.columns 
    assert 'cluster' in result.columns 
    assert result['sales'].iloc[0] == 100 and result['sales'].iloc[2] == 300
    assert result.loc[result['store_nbr'] == 1, 'city'].values[0] == 'Quito'
    with pytest.raises(pd.errors.MergeError): 
        merge_additional_data(X, data_duplicate, on='store_nbr')