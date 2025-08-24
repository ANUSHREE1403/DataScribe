import joblib

def save_model(model, path):
    joblib.dump(model, path)

def infer_task_type(df, target_col):
    n_unique = df[target_col].nunique()
    if df[target_col].dtype == 'object' or n_unique < 20:
        return 'classification'
    else:
        return 'regression' 