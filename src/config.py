import yaml 

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

def validate_config(config): 
    required_keys = ['project', 'data', 'features', 'model', 'training', 'tracking']
    missing_keys = [] 
    not_valid_values = []
    for key in required_keys: 
        if key not in config: 
            missing_keys.append(key) 
        else:
            if not isinstance(config[key], dict): 
                not_valid_values.append(key)
    if missing_keys: 
        raise ValueError(f'Config is not correct! Missing keys: {missing_keys}')
    if not_valid_values: 
        raise ValueError(f"Values for these keys: {not_valid_values} are not correct!")