import yaml 


REQUIRED_NESTED_KEYS = {
    'project': ['name', 'random_state'],
    'data': ['train_path', 'test_path', 'data_dir', 'validation_start'],
    'features': ['group_columns', 'horizon', 'lags', 'rolling_windows'],
    'model': ['name', 'params'],
    'training': ['early_stopping_rounds', 'eval_metric'],
    'tracking': ['experiment_name', 'tracking_uri', 'log_model'],
    'backtesting': ['initial_train_end', 'validation_days', 'n_folds'],
    'hpo': ['n_trials', 'direction', 'metric']}

REQUIRED_KEYS = ['project', 'data', 'features', 
                 'model', 'training', 'tracking', 
                 'backtesting', 'hpo']

def load_config(config_path): 
    if not config_path.exists(): 
        raise FileNotFoundError(f"Config path is not correct: {config_path}!")
    with open(config_path, "r") as f: 
        config = yaml.safe_load(f) 
    if config is None: 
        raise ValueError("Config file is empty!")
    if not isinstance(config, dict):
        raise ValueError('Config should be dictionary!')
    return config 


def validate_section(section_name, section): 
    missing_keys = [key for key in REQUIRED_NESTED_KEYS[section_name]
                    if key not in section]
    if missing_keys:
        raise ValueError(f"Missing keys into section '{section_name}': {missing_keys}")


def validate_values(config):
    if config['features']['horizon'] <= 0: 
        raise ValueError("features.horizon must be greater than 0")
    if config['backtesting']['n_folds'] <= 0: 
        raise ValueError("backtesting.n_folds must be greater than 0")
    if config['backtesting']['validation_days'] != config['features']['horizon']: 
        raise ValueError("backtesting.validation_days must be equal to features.horizon")
    if config['hpo']['direction'] != 'minimize':
        raise ValueError("hpo.direction must be 'minimize'")


def validate_config(config): 
    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing config section: {missing_keys}")
    invalid_sections = [key for key in REQUIRED_KEYS if not isinstance(config[key], dict)]
    if invalid_sections:
        raise ValueError(f"Config sections must be dictionaries: {invalid_sections}")
    for key in REQUIRED_KEYS:
        validate_section(key, config[key])
    validate_values(config)
    