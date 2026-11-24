import mlflow 
import json 
from functools import wraps

def _serialize_config_params(config, prefix): 
    return {
        f"{prefix}.{key}": json.dumps(value) if isinstance(value, list) else value 
        for key, value in config.items()
    }

def setup_tracking(tracking_config): 
    mlflow.set_tracking_uri(tracking_config['tracking_uri'])
    mlflow.set_experiment(tracking_config['experiment_name'])

def log_run_metadata(git_commit, git_dirty, model_name): 
    mlflow.set_tag("git_commit", git_commit) 
    mlflow.set_tag("git_dirty", str(git_dirty).lower())
    mlflow.set_tag("model_name", model_name) 

def log_training_results(params_config, feature_config, training_config, metrics): 
    mlflow.log_metric("RMSLE", metrics['rmsle'])
    mlflow.log_metric("MAE", metrics['mae']) 
    mlflow.log_metric("best_iteration", metrics['best_iteration'])
    feature_params = _serialize_config_params(feature_config, "features")
    training_params = _serialize_config_params(training_config, "training")
    model_params = _serialize_config_params(params_config, "model")
    mlflow.log_params(model_params)
    mlflow.log_params(feature_params)
    mlflow.log_params(training_params) 

def log_model_artifact(model, config): 
    mlflow.lightgbm.log_model(model, name="lightgbm-model")
    mlflow.log_dict(config, "config.json")

def log_backtesting_results(model_params, feature_config, training_config, backtesting_config, results_df, summary): 
    mlflow.log_params(_serialize_config_params(model_params, "model"))
    mlflow.log_params(_serialize_config_params(feature_config, "features"))
    mlflow.log_params(_serialize_config_params(training_config, "training"))
    mlflow.log_params(_serialize_config_params(backtesting_config, "backtesting"))
    mlflow.log_metric("mean_rmsle", results_df["rmsle"].mean())
    mlflow.log_metric("mean_mae", results_df["mae"].mean())
    mlflow.log_metric("std_rmsle", results_df["rmsle"].std())
    mlflow.log_metric("std_mae", results_df["mae"].std()) 
    for idx, row in results_df.iterrows(): 
        fold_number = idx + 1 
        mlflow.log_metric(f"fold_{fold_number}_rmsle", row['rmsle'])
        mlflow.log_metric(f"fold_{fold_number}_mae", row['mae'])
    mlflow.log_dict(summary.to_dict(), "backtesting_summary.json")
    results_df.to_csv("backtesting_results.csv", index=False)
    mlflow.log_artifact("backtesting_results.csv")

def mlflow_run(func): 
    @wraps(func)
    def wrapper(*args, **kwargs): 
        config = args[0] if args else kwargs.get("config")
        if config is None: 
            raise ValueError("Missing config argument in decorated function!")
        setup_tracking(config['tracking'])
        with mlflow.start_run(): 
            results = func(*args, **kwargs)
        return results
    return wrapper

def log_hpo_trial(study, trial): 
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("mae", trial.value)
        mlflow.log_metric("rmsle", trial.user_attrs["rmsle"])
        mlflow.log_metric("best_iteration", trial.user_attrs["best_iteration"])
        
