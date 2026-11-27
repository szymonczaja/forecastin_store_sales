import yaml 


REQUIRED_NESTED_KEYS = {
    'project': ['name', 'random_state'],
    'data': ['train_path', 'test_path', 'data_dir', 'validation_start'],
    'features': ['group_columns', 'horizon', 'lags', 'rolling_windows', 'rolling_std_window'],
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
    features = config['features']
    backtesting = config['backtesting']
    training = config['training']
    hpo = config['hpo']
    integer_values = {
        'features.horizon': features['horizon'],
        'backtesting.n_folds': backtesting['n_folds'],
        'backtesting.validation_days': backtesting['validation_days'],
        'hpo.n_trials': hpo['n_trials'],
        'training.early_stopping_rounds': training['early_stopping_rounds'], 
        'features.rolling_std_window': features['rolling_std_window'],
    }
    for name, value in integer_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")
    list_values = {
        'features.lags': features['lags'],
        'features.rolling_windows': features['rolling_windows'],
        'features.group_columns': features['group_columns']
    }
    for name, value in list_values.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a non-empty list")
    for name in ['lags', 'rolling_windows']:
        values = features[name]
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in values
        ):
            raise ValueError(f"features.{name} must contain positive integers")
    if not all(isinstance(value, str) and value for value in features['group_columns']):
        raise ValueError("features.group_columns must contain non-empty strings")
    if not isinstance(config['model']['params'], dict):
        raise ValueError("model.params must be a dictionary")
    string_values = {
        'model.name': config['model']['name'],
        'training.eval_metric': training['eval_metric'],
        'tracking.experiment_name': config['tracking']['experiment_name'],
        'tracking.tracking_uri': config['tracking']['tracking_uri'],
        'hpo.direction': hpo['direction'],
        'hpo.metric': hpo['metric']
    }
    for name, value in string_values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(config['tracking']['log_model'], bool):
        raise ValueError("tracking.log_model must be a boolean")
    if backtesting['validation_days'] != features['horizon']:
        raise ValueError("backtesting.validation_days must be equal to features.horizon")
    if hpo['direction'] != 'minimize':
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
    