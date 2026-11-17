import argparse 
from src.config import load_config, validate_config
from src.preprocessing import load_raw_data, prepare_and_split_data
from src.train import train_model, predict, calc_rmsle
from pathlib import Path
import mlflow 


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    mlflow.set_tracking_uri(tracking_config['tracking_uri'])
    mlflow.set_experiment(tracking_config['experiment_name'])
    with mlflow.start_run():
        model = train_model(X_train, y_train, X_val, y_val,
                            params_config, early_stopping_rounds,
                            eval_metric)
        y_pred = predict(model, X_val) 
        rmsle = calc_rmsle(y_val, y_pred) 
        mlflow.log_params(params_config) 
        mlflow.log_metric("RMSLE", rmsle) 
        mlflow.log_metric("best_iteration", model.best_iteration_)
        mlflow.lightgbm.log_model(model, name="lightgbm-model") 
        mlflow.log_dict(config, "config.json")
    print(f"Model best iteration: {model.best_iteration_}")
    print(f"Model score RMSLE: {rmsle:.4f}")
    

if __name__ == "__main__": 
    main()