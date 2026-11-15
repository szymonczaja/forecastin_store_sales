import argparse 
from src.config import load_config, validate_config
from pathlib import Path

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


if __name__ == "__main__": 
    main()