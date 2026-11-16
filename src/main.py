import argparse 
from src.config import load_config, validate_config
from src.preprocessing import load_raw_data, prepare_and_split_data
from pathlib import Path

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

if __name__ == "__main__": 
    main()