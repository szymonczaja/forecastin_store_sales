from preprocessing import load_raw_data, prepare_and_split_data
import pandas as pd 
import numpy as np 
import lightgbm as lgb 

LGBM_PARAMS = {
            'objective' : 'regression',
            'n_estimators' : 1500, 
            'learning_rate' : 0.05, 
            'num_leaves' : 31, 
            'max_depth' : -1, 
            'random_state' : 42,
            'n_jobs' : -1, 
            'verbose' : -1
            }

def calc_rmsle(y_true, y_pred) -> float: 
    if len(y_pred) != len(y_true): 
        raise ValueError(f"Length of y_pred and y_true are not the same! True: {len(y_true)} | Pred: {len(y_pred)}")
    y_true = np.asarray(y_true) 
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)))

def train_model(X_train, y_train, X_val, y_val): 
    model = lgb.LGBMRegressor(**LGBM_PARAMS) 
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    model.fit(X_train, y_train_log,
              eval_metric='rmse',
              eval_set=[(X_val, y_val_log)], 
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model

def predict(model, X): 
    y_pred = model.predict(X) 
    y_pred = np.expm1(y_pred) 
    return np.clip(y_pred, 0, None) 

def main(): 
    train, test = load_raw_data() 
    X_train, y_train, X_val, y_val, X_test, test_id = prepare_and_split_data(train, test)
    model = train_model(X_train, y_train, X_val, y_val) 
    y_pred = predict(model, X_val) 
    rmsle = calc_rmsle(y_val, y_pred) 
    print(f"Model best iteration: {model.best_iteration_}")
    print(f"Model score: RMSLE: {rmsle:.4f}")

if __name__ == '__main__': 
    main()
