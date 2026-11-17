import numpy as np 
import lightgbm as lgb 

def calc_rmsle(y_true, y_pred) -> float: 
    if len(y_pred) != len(y_true): 
        raise ValueError(f"Length of y_pred and y_true are not the same! True: {len(y_true)} | Pred: {len(y_pred)}")
    y_true = np.asarray(y_true) 
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)))

def train_model(X_train, y_train, X_val, y_val, params_config, early_stopping_rounds, eval_metric_config): 
    model = lgb.LGBMRegressor(**params_config) 
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    model.fit(X_train, y_train_log,
              eval_metric=eval_metric_config,
              eval_set=[(X_val, y_val_log)], 
              callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    return model

def predict(model, X): 
    y_pred = model.predict(X) 
    y_pred = np.expm1(y_pred) 
    return np.clip(y_pred, 0, None) 


