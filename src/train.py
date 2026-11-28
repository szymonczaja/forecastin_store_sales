import numpy as np 
import lightgbm as lgb 

class ModelTrainer:
    def __init__(self, model_params, early_stopping_rounds, eval_metric): 
        self.model_params = model_params.copy() 
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric 
        self.model = None 

    @staticmethod
    def _transform_targets(y_train, y_val):
        return np.log1p(y_train), np.log1p(y_val)

    @property
    def best_iteration(self) -> int | None: 
        if self.model is None: 
            raise RuntimeError("Model is not fitted yet.")
        return self.model.best_iteration_

    def fit(self, X_train, y_train, X_val, y_val) -> "ModelTrainer": 
        model = lgb.LGBMRegressor(**self.model_params)
        y_train_log, y_val_log = self._transform_targets(y_train, y_val)
        model.fit(X_train, y_train_log, 
                  eval_metric=self.eval_metric, 
                  eval_set=[(X_val, y_val_log)],
                  callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        self.model = model 
        return self

    def predict(self, X) -> np.ndarray: 
        if self.model is None: 
            raise RuntimeError("Model is not fitted yet.")
        y_pred_log = self.model.predict(X) 
        y_pred = np.expm1(y_pred_log) 
        return np.clip(np.asarray(y_pred), 0, None)