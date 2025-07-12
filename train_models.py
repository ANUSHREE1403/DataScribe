from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    has_lightgbm = True
except ImportError:
    has_lightgbm = False
from tqdm import tqdm

def train_and_return_models(X_train, y_train, task_type, selected_models=None):
    models = {}
    all_model_defs = {}
    if task_type == 'classification':
        all_model_defs = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(eval_metric='logloss'),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True),
            'DecisionTree': DecisionTreeClassifier(),
        }
        if has_lightgbm:
            all_model_defs['LightGBM'] = LGBMClassifier()
    else:
        all_model_defs = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'KNN': KNeighborsRegressor(),
            'SVR': SVR(),
            'DecisionTree': DecisionTreeRegressor(),
        }
        if has_lightgbm:
            all_model_defs['LightGBM'] = LGBMRegressor()
    # Filter by selected_models if provided
    if selected_models:
        model_defs = [(name, all_model_defs[name]) for name in selected_models if name in all_model_defs]
    else:
        model_defs = list(all_model_defs.items())
    for name, model in tqdm(model_defs, desc='Training models'):
        models[name] = model.fit(X_train, y_train)
    return models 