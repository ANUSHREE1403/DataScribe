from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_models(models, X_test, y_test, task_type):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if task_type == 'classification':
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) == 2 else None
            }
        else:
            results[name] = {
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': mean_squared_error(y_test, y_pred, squared=False)
            }
    return results

def print_model_comparison(results, task_type):
    import pandas as pd
    df = pd.DataFrame(results).T
    if task_type == 'classification':
        sort_col = 'F1'
    else:
        sort_col = 'R2'
    df_sorted = df.sort_values(by=sort_col, ascending=False)
    print("\nModel Comparison:")
    print(df_sorted) 