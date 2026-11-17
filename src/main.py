import argparse 
from src.config import load_config, validate_config
from src.preprocessing import load_raw_data, prepare_and_split_data
from src.train import train_model, predict, calc_rmsle
from pathlib import Path
import time 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main(): 
    start = time.perf_counter()
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
    model = train_model(X_train, y_train, X_val, y_val,
                         params_config, early_stopping_rounds,
                         eval_metric)
    y_pred = predict(model, X_val) 
    rmsle = calc_rmsle(y_val, y_pred) 
    end = time.perf_counter() 
    print(f"Model best iteration: {model.best_iteration_}")
    print(f"Model score RMSLE: {rmsle:.4f}")
    print(f"Czas działania: {end - start}")

if __name__ == "__main__": 
    main()