"""
Définitions des modèles ML
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import config

def get_models():
    """Retourne le dictionnaire de tous les modèles"""
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=1000,
            solver='liblinear'
        ),
        
        'RandomForest': RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        
        'XGBoost': XGBClassifier(
            random_state=config.RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        
        'LightGBM': LGBMClassifier(
            random_state=config.RANDOM_STATE,
            verbose=-1
        )
    }
    
    return models

def get_param_grids():
    """Retourne les grilles de paramètres pour GridSearch"""
    return config.PARAM_GRIDS

def get_model_by_name(model_name):
    """Récupère un modèle spécifique par son nom"""
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Modèle '{model_name}' non reconnu. Disponibles: {list(models.keys())}")
    return models[model_name]

def get_best_params(model_name):
    """Retourne les meilleurs hyperparamètres trouvés (à mettre à jour après tuning)"""
    # Ces valeurs seront mise à jour après GridSearchCV
    best_params = {
        'LogisticRegression': {
            'C': 1.0,
            'max_iter': 1000
        },
        'RandomForest': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 2
        },
        'XGBoost': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8
        },
        'LightGBM': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_leaves': 31
        }
    }
    
    return best_params.get(model_name, {})
