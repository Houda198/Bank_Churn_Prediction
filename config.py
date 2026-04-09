"""
Configuration centralisée du projet
"""

import os
from pathlib import Path

# ========== PATHS ==========
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Créer les dossiers s'ils n'existent pas
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ========== DATA ==========
RAW_DATA_FILE = RAW_DATA_DIR / "Churn_Modelling.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "churn_processed.csv"

# ========== MODEL ==========
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Features à exclure
FEATURES_TO_DROP = ['RowNumber', 'CustomerId', 'Surname']

# Target
TARGET_COL = 'Exited'

# Colonnes catégorielles
CATEGORICAL_FEATURES = ['Geography', 'Gender']

# Colonnes numériques
NUMERICAL_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                     'EstimatedSalary']

# ========== MODELING ==========
# Paramètres pour gestion du déséquilibre
SMOTE_SAMPLING_STRATEGY = 0.5  # Ratio minorité/majorité après SMOTE

# Modèles à tester
MODELS_TO_TEST = [
    'LogisticRegression',
    'RandomForest', 
    'XGBoost',
    'LightGBM'
]

# Hyperparamètres pour GridSearch
PARAM_GRIDS = {
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }
}

# ========== VISUALIZATION ==========
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_PALETTE = 'husl'
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'

# Couleurs personnalisées
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'dark': '#2c3e50'
}

# ========== LOGGING ==========
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
