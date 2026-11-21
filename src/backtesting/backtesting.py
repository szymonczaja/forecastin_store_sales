import pandas as pd 
from src.train import train_model, predict, calc_mae, calc_rmsle
from src.preprocessing import preprocessor, convert_object_to_category


def generate_time_splits(df, initial_train_end, validation_days) -> list: 
    """Generate temporal train/validation folds based on dates."""
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
    return folds


def split_fold_data(df, fold, feature_config):
    train_start = fold['train_start']
    train_end = fold['train_end']
    val_start = fold['val_start']
    val_end = fold['val_end']
    fold_df = df[df['date'] <= val_end].copy()
    preprocessed_df = convert_object_to_category(preprocessor(fold_df, feature_config))
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


def run_backtesting(df, folds, model_params, early_stopping_rounds, eval_metric, feature_config): 
    """Run backtest for all folds"""
    results = [] 
    for fold in folds: 
        X_train, y_train, X_val, y_val = split_fold_data(df, fold, feature_config)
        model = train_model(X_train, y_train, X_val, y_val, model_params, early_stopping_rounds, eval_metric)
        y_pred = predict(model, X_val)
        rmsle = calc_rmsle(y_val, y_pred) 
        mae = calc_mae(y_val, y_pred) 
        fold_res = {'train_start' : fold['train_start'], 
                    'train_end' : fold['train_end'],
                    'val_start': fold['val_start'], 
                    'val_end' : fold['val_end'],
                    'rmsle' : rmsle,
                    'mae' : mae, 
                    'best_iteration' : model.best_iteration_
                    }
        results.append(fold_res)
    return results


def aggregate_results(results): 
    results_df = pd.DataFrame(results)
    summary = results_df[['rmsle', 'mae']].describe() 
    return results_df, summary 
