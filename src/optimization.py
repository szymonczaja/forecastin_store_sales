from src.evaluate import calc_mae, calc_rmsle 
from src.train import ModelTrainer


def objective(trial, X_train, y_train, X_val, y_val, params_config, early_stopping_rounds, eval_metric) -> float:
    base_params = params_config.copy()
    base_params.update({
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
    trainer = ModelTrainer(model_params=base_params, 
                           early_stopping_rounds=early_stopping_rounds, 
                           eval_metric=eval_metric)
    trainer.fit(X_train, y_train, X_val, y_val)
    pred = trainer.predict(X_val)
    mae = calc_mae(y_val, pred) 
    rmsle = calc_rmsle(y_val, pred) 
    trial.set_user_attr("rmsle", rmsle)
    trial.set_user_attr("best_iteration", trainer.best_iteration)
    return mae