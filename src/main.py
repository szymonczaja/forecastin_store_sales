import argparse 
import subprocess
from pathlib import Path
import mlflow
from src.config import load_config, validate_config
from src.preprocessing import load_raw_data, prepare_and_split_data
from src.train import train_model, predict, calc_rmsle, calc_mae
from src.tracking.mlflow_tracker import setup_tracking, log_run_metadata, log_training_results, log_model_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_git_info() -> tuple[str, bool]: 
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], 
        cwd=PROJECT_ROOT, 
        capture_output=True, 
        text=True, 
        check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ['git', 'status', '--porcelain'], 
        cwd=PROJECT_ROOT, 
        capture_output=True, 
        text=True, 
        check=True, 
    ).stdout.strip() != ""
    return commit, dirty 


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--config", 
        type=str, 
        required=True
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path) 
    validate_config(config)
    train_path = Path(PROJECT_ROOT / config['data']['train_path'])
    test_path = Path(PROJECT_ROOT / config['data']['test_path'])
    train_df, test_df = load_raw_data(train_path, test_path)
    X_train, y_train, X_val, y_val, X_test, test_ids = prepare_and_split_data(train_df, test_df, config)
    params_config = config['model']['params']
    early_stopping_rounds = config['training']['early_stopping_rounds']
    eval_metric = config['training']['eval_metric']
    tracking_config = config['tracking']
    model_config = config['model']
    features_config = config['features']
    training_config = config['training']
    git_commit, git_dirty = get_git_info()
    setup_tracking(tracking_config)
    with mlflow.start_run():
        log_run_metadata(git_commit, git_dirty, model_config['name'])
        model = train_model(X_train, y_train, X_val, y_val,
                            params_config, early_stopping_rounds,
                            eval_metric)
        y_pred = predict(model, X_val) 
        rmsle = calc_rmsle(y_val, y_pred) 
        mae = calc_mae(y_val, y_pred)
        best_iter = model.best_iteration_
        metrics = {'rmsle':rmsle, 'mae':mae, 'best_iter':best_iter}
        log_training_results(params_config, features_config, training_config, metrics)
        log_model_artifact(model, config)
    print(f"Model best iteration: {model.best_iteration_}")
    print(f"Model score RMSLE: {rmsle:.4f}")
    
if __name__ == "__main__": 
    main()