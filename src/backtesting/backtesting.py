import pandas as pd
from src.train import ModelTrainer
from src.evaluate import calc_mae, calc_rmsle
from src.preprocessing.preprocessing import preprocessor, convert_object_to_category
from src.preprocessing.external_data import merge_external_data


def generate_time_splits(df, initial_train_end, validation_days, n_folds) -> list[dict]: 
    """Generate temporal train/validation folds based on dates."""
    if validation_days <= 0: 
        raise ValueError("Validation days must be greater than 0")
    if n_folds <= 0: 
        raise ValueError("n_folds must be greater than 0")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("'date' column must have datetime dtype.")
    if df.empty:
        raise ValueError("Input DataFrame is empty!")
    initial_train_end = pd.Timestamp(initial_train_end)
    train_end = initial_train_end
    val_start = train_end + pd.Timedelta(days=1) 
    val_end = val_start + pd.Timedelta(days=validation_days - 1) 
    train_start = df['date'].min()
    folds = []
    while val_end <= df['date'].max(): 
        fold = {
            'train_start' : train_start, 
            'train_end' : train_end, 
            'val_start' : val_start, 
            'val_end' : val_end
        }
        folds.append(fold) 
        train_end = val_end 
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=validation_days - 1)
    if len(folds) < n_folds: 
        raise ValueError(f"Requested {n_folds}, but only {len(folds)} could be generated.")
    return folds[-n_folds:]


def prepare_fold_data(fold_df: pd.DataFrame, external_data: dict, feature_config: dict) -> pd.DataFrame: 
    merged_df = merge_external_data(fold_df, external_data, feature_config['lags'], feature_config['rolling_windows'], feature_config['horizon'])
    preprocessed_df = preprocessor(merged_df, feature_config)
    return convert_object_to_category(preprocessed_df)  


def split_fold_data(df, fold, external_data, feature_config):
    train_start = fold['train_start']
    train_end = fold['train_end']
    val_start = fold['val_start']
    val_end = fold['val_end']
    fold_df = df[df['date'] <= val_end].copy()
    preprocessed_df = prepare_fold_data(fold_df, external_data, feature_config)
    train_mask = ((preprocessed_df['date'] >= train_start) & (preprocessed_df['date'] <= train_end))
    val_mask = ((preprocessed_df['date'] >= val_start) & (preprocessed_df['date'] <= val_end))
    train_df = preprocessed_df[train_mask]
    val_df = preprocessed_df[val_mask] 
    drop_cols = ['id', 'sales', 'date']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['sales']
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df['sales']
    return X_train, y_train, X_val, y_val 

class Backtester: 
    def __init__(self, model_params, early_stopping_rounds, eval_metric, 
                 feature_config, external_data): 
        self.model_params = model_params.copy()
        self.early_stopping_rounds = early_stopping_rounds 
        self.eval_metric = eval_metric 
        self.feature_config = feature_config.copy()
        self.external_data = external_data 

    def _run_fold(self, df, fold): 
        X_train, y_train, X_val, y_val = split_fold_data(df, fold, self.external_data, self.feature_config)
        trainer = ModelTrainer(self.model_params, self.early_stopping_rounds, self.eval_metric)
        trainer.fit(X_train, y_train, X_val, y_val)
        y_pred = trainer.predict(X_val) 
        rmsle = calc_rmsle(y_val, y_pred) 
        mae = calc_mae(y_val, y_pred) 
        fold_result = {'train_start' : fold['train_start'], 
                        'train_end' : fold['train_end'],
                        'val_start': fold['val_start'], 
                        'val_end' : fold['val_end'],
                        'rmsle' : rmsle,
                        'mae' : mae, 
                        'best_iteration' : trainer.best_iteration
                        }
        return fold_result

    def run(self, df, folds): 
        return [self._run_fold(df, fold) for fold in folds]


def aggregate_results(results): 
    if not results: 
        raise ValueError("results list is empty.")
    results_df = pd.DataFrame(results)
    summary = results_df[['rmsle', 'mae']].describe() 
    return results_df, summary 