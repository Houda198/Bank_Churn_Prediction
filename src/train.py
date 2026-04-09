"""
Module d'entraînement des modèles (Version Robuste - Sans SMOTE)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import joblib
import config
import matplotlib.pyplot as plt


def load_processed_data(filepath=None):
    """Charge les données nettoyées"""
    if filepath is None:
        filepath = config.PROCESSED_DATA_FILE
    
    print(f"📂 Chargement: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ {len(df):,} lignes chargées")
    return df

def prepare_train_val_test_split(df):
    """Split propre: Train / Validation / Test"""
    print("\n🔀 Split Train/Val/Test...")
    
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    
    # Split 1: Train+Val (80%) vs Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Split 2: Train (64%) vs Val (16%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"✓ Train: {len(X_train):,} ({y_train.mean()*100:.1f}% churn)")
    print(f"✓ Val:   {len(X_val):,} ({y_val.mean()*100:.1f}% churn)")
    print(f"✓ Test:  {len(X_test):,} ({y_test.mean()*100:.1f}% churn)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_features(X_train, X_val, X_test):
    """Normalisation sur train, application sur val/test"""
    print("\n📏 Normalisation...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconvertir en DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Sauvegarder scaler
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler sauvegardé")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def get_optimized_models():
    """Modèles avec class_weight et régularisation"""
    
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            solver='liblinear'
        ),
        
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.RANDOM_STATE,
            verbose=-1
        )
    }
    
    return models

def train_with_validation(model_name, model, X_train, y_train, X_val, y_val):
    """Entraîne avec monitoring validation"""
    print(f"\n🎯 {model_name}")
    
    # Scale pos weight pour tree-based
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # Entraînement
    if model_name in ['XGBoost', 'LightGBM']:
        if model_name == 'XGBoost':
            model.set_params(scale_pos_weight=scale_pos_weight)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            model.set_params(scale_pos_weight=scale_pos_weight)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[]
            )
    else:
        model.fit(X_train, y_train)
    
    # Métriques
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_recall = recall_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    
    gap = train_f1 - val_f1
    
    print(f"  Train:  F1={train_f1:.3f} AUC={train_auc:.3f}")
    print(f"  Val:    F1={val_f1:.3f} AUC={val_auc:.3f}")
    print(f"  Gap:    {gap:.3f} {'⚠️' if gap > 0.05 else '✓'}")
    
    return model, {
        'train_f1': train_f1, 'train_auc': train_auc,
        'val_f1': val_f1, 'val_auc': val_auc,
        'val_recall': val_recall, 'val_precision': val_precision,
        'gap': gap
    }

def train_all_models(X_train, y_train, X_val, y_val):
    """Entraîne tous les modèles"""
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT (Class Weights - Pas de SMOTE)")
    print("="*70)
    
    models_dict = get_optimized_models()
    trained_models = {}
    all_metrics = {}
    
    for name, model in models_dict.items():
        trained_model, metrics = train_with_validation(
            name, model, X_train, y_train, X_val, y_val
        )
        trained_models[name] = trained_model
        all_metrics[name] = metrics
    
    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ")
    print("="*70)
    
    df = pd.DataFrame(all_metrics).T
    df = df.sort_values('val_f1', ascending=False)
    print(df[['val_f1', 'val_recall', 'val_precision', 'gap']].to_string())
    
    # Overfitting check
    print("\n⚠️  Overfitting (gap > 0.05):")
    bad = df[df['gap'] > 0.05]
    if len(bad) > 0:
        for idx in bad.index:
            print(f"  - {idx}: gap={bad.loc[idx, 'gap']:.3f}")
    else:
        print("  ✓ Aucun!")
    
    return trained_models, df

def save_models(models):
    """Sauvegarde modèles"""
    print(f"\n💾 Sauvegarde...")
    for name, model in models.items():
        joblib.dump(model, config.MODELS_DIR / f"{name}.pkl")
        print(f"✓ {name}")

def main(use_grid_search=False):
    """Pipeline principal"""
    print("\n" + "="*70)
    print("🤖 PIPELINE ENTRAÎNEMENT (Robuste)")
    print("="*70)
    
    df = load_processed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(df)
    X_train_s, X_val_s, X_test_s, scaler = normalize_features(X_train, X_val, X_test)
    
    trained_models, results_df = train_all_models(X_train_s, y_train, X_val_s, y_val)
    
    save_models(trained_models)
    
    # Sauvegarder test/val data
    test_data = {
        'X_test': X_test_s, 'y_test': y_test,
        'X_val': X_val_s, 'y_val': y_val
    }
    joblib.dump(test_data, config.MODELS_DIR / "test_data.pkl")
    print("✓ Test/val data sauvegardées")
    
    print("\n✨ Terminé!")
    
    return trained_models, results_df, X_test_s, y_test

def plot_importance(model, features):
    # Récupérer les importances
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 8))
    plt.title('Variables les plus importantes (Random Forest)')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importance relative')
    plt.show()


if __name__ == "__main__":
    # On récupère ce que main() retourne
    trained_models, results_df, X_test_s, y_test = main()
    
    # On appelle la fonction avec le meilleur modèle et les noms de colonnes
    plot_importance(trained_models['RandomForest'], X_test_s.columns)
