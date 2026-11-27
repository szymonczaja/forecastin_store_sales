import argparse 
import subprocess
from pathlib import Path
import optuna
import mlflow
from functools import partial
from src.config import load_config, validate_config
from src.preprocessing.preprocessing import load_raw_data, prepare_and_split_data
from src.train import train_model, predict, calc_rmsle, calc_mae, objective
from src.tracking.mlflow_tracker import log_run_metadata, log_training_results, log_model_artifact, mlflow_run, log_backtesting_results, log_hpo_trial
from src.backtesting.backtesting import run_backtesting, generate_time_splits, aggregate_results
from src.preprocessing.external_data import load_external_data 

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


@mlflow_run
def run_training(config, git_commit, git_dirty, X_train, y_train, X_val, y_val): 
    params_config = config['model']['params']
    features_config = config['features']
    training_config = config['training']
    model_config = config['model']
    early_stopping_rounds = training_config['early_stopping_rounds']
    eval_metric = training_config['eval_metric']
    log_run_metadata(git_commit, git_dirty, model_config['name'])
    model = train_model(X_train, y_train, X_val, y_val,
                            params_config, early_stopping_rounds,
                            eval_metric)
    y_pred = predict(model, X_val) 
    rmsle = calc_rmsle(y_val, y_pred) 
    mae = calc_mae(y_val, y_pred)
    best_iteration = model.best_iteration_
    metrics = {'rmsle':rmsle, 'mae':mae, 'best_iteration':best_iteration}
    log_training_results(params_config, features_config, training_config, metrics)
    log_model_artifact(model, config)    
    return model, metrics 


@mlflow_run
def run_backtest(config, git_commit, git_dirty, df):
    model_config = config['model']
    external_data = load_external_data(config['data']['data_dir'])
    features_config = config['features']
    params_config = config['model']['params']
    training_config = config['training']
    backtesting_config = config['backtesting']
    eval_metric = training_config['eval_metric']
    early_stopping_rounds = training_config['early_stopping_rounds']
    initial_train_end = backtesting_config['initial_train_end']
    validation_days = backtesting_config['validation_days'] 
    n_folds = backtesting_config['n_folds'] 
    if validation_days != features_config['horizon']:
        raise ValueError(f"validation_days ({validation_days}) must equal forecast horizon ({features_config['horizon']})")
    folds = generate_time_splits(df, initial_train_end, validation_days, n_folds)
    backtesting_results = run_backtesting(df, folds, params_config, early_stopping_rounds, eval_metric, features_config, external_data)
    results_df, summary = aggregate_results(backtesting_results)
    log_backtesting_results(
        params_config,
        features_config,
        training_config,
        backtesting_config,
        results_df,
        summary
    )
    log_run_metadata(git_commit, git_dirty, f"{model_config['name']}_backtest")
    return results_df, summary

def run_hpo(config, git_commit, git_dirty, X_train, y_train, X_val, y_val):
    params_config = config['model']['params'] 
    early_stopping_rounds = config['training']['early_stopping_rounds']
    eval_metric_config = config['training']['eval_metric']
    hpo_config = config['hpo']
    n_trials = hpo_config['n_trials']
    direction = hpo_config['direction']
    mlflow.set_tracking_uri(config['tracking']['tracking_uri'])
    mlflow.set_experiment(config['tracking']['experiment_name'])
    with mlflow.start_run(run_name="hpo"):
        objective_func = partial(
            objective, 
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val, 
            params_config=params_config, 
            early_stopping_rounds=early_stopping_rounds, 
            eval_metric_config=eval_metric_config
        )
        study = optuna.create_study(direction=direction)
        study.optimize(objective_func, n_trials=n_trials, callbacks=[log_hpo_trial])
        best_params = study.best_params 
        best_score = study.best_value
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("direction", direction)
        mlflow.log_metric("best_mae", best_score)
        mlflow.log_dict(best_params, "best_params.json")
        log_run_metadata(git_commit, git_dirty, f"{config['model']['name']}_hpo")
    return {
        "best_params": best_params, 
        "best_mae": best_score, 
        "n_trials" : n_trials
    }


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--config", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--mode",
        type=str, 
        choices=['train', 'backtest', 'hpo'], 
        default="train"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path) 
    validate_config(config)
    git_commit, git_dirty = get_git_info()
    train_path = Path(PROJECT_ROOT / config["data"]["train_path"])
    if args.mode in ["train", "hpo"]: 
        test_path = Path(PROJECT_ROOT / config["data"]["test_path"])
        train_df, test_df = load_raw_data(train_path, test_path)
        X_train, y_train, X_val, y_val, X_test, test_ids = prepare_and_split_data(train_df, test_df, config)
        if args.mode == "train":
            run_training(config, git_commit, git_dirty, X_train, y_train, X_val, y_val)
        elif args.mode == "hpo":
            results=run_hpo(config, git_commit, git_dirty, X_train, y_train, X_val, y_val)
            print(results)
    elif args.mode == "backtest": 
        train_df = load_raw_data(train_path) 
        run_backtest(config, git_commit, git_dirty, train_df)
    
    
if __name__ == "__main__": 
    main()