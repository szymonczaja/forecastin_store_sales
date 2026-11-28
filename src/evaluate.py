import numpy as np 
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error

def calc_rmsle(y_true, y_pred) -> float: 
    return root_mean_squared_log_error(y_true, y_pred) 

def calc_mae(y_true, y_pred) -> float: 
    return mean_absolute_error(y_true, y_pred)
