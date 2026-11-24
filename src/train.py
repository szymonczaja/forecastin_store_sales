import numpy as np 
import lightgbm as lgb 
import optuna
from sklearn.metrics import mean_absolute_error

def calc_rmsle(y_true, y_pred) -> float: 
    if len(y_pred) != len(y_true): 
        raise ValueError(f"Length of y_pred and y_true are not the same! True: {len(y_true)} | Pred: {len(y_pred)}")
    y_true = np.asarray(y_true) 
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)))

def calc_mae(y_true, y_pred) -> float: 
    return mean_absolute_error(y_true, y_pred)

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


def objective(trial, X_train, y_train, X_val, y_val, params_config, early_stopping_rounds, eval_metric_config) -> float:
    params = params_config.copy()
    params.update({
    "learning_rate": trial.suggest_float(
        "learning_rate", 0.01, 0.1, log=True
    ),
    "num_leaves": trial.suggest_int(
        "num_leaves", 20, 150
    ),
    "max_depth": trial.suggest_int(
        "max_depth", 4, 12
    ),
    "min_child_samples": trial.suggest_int(
        "min_child_samples", 10, 100
    ),
    "subsample": trial.suggest_float(
        "subsample", 0.6, 1.0
    ),
    "colsample_bytree": trial.suggest_float(
        "colsample_bytree", 0.6, 1.0
    ),
    "reg_alpha": trial.suggest_float(
        "reg_alpha", 1e-8, 10.0, log=True
    ),
    "reg_lambda": trial.suggest_float(
        "reg_lambda", 1e-8, 10.0, log=True
    ),
    })
    model = train_model(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        params, 
        early_stopping_rounds, 
        eval_metric_config
    )
    pred = predict(model, X_val)
    mae = calc_mae(y_val, pred) 
    rmsle = calc_rmsle(y_val, pred) 
    trial.set_user_attr("rmsle", rmsle)
    trial.set_user_attr("best_iteration", model.best_iteration_)
    return mae

