import mlflow 
import json 

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
    mlflow.log_metric("best_iteration", metrics['best_iter'])
    feature_params = _serialize_config_params(feature_config, "features")
    training_params = _serialize_config_params(training_config, "training")
    model_params = _serialize_config_params(params_config, "model")
    mlflow.log_params(model_params)
    mlflow.log_params(feature_params)
    mlflow.log_params(training_params) 

def log_model_artifact(model, config): 
    mlflow.lightgbm.log_model(model, name="lightgbm-model")
    mlflow.log_dict(config, "config.json")
