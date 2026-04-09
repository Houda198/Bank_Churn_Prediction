"""
Module de nettoyage des données
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config

def load_raw_data(filepath=None):
    """Charge les données brutes"""
    if filepath is None:
        filepath = config.RAW_DATA_FILE
    
    print(f"📂 Chargement des données depuis {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ {len(df):,} lignes et {len(df.columns)} colonnes chargées")
    return df

def check_data_quality(df):
    """Vérifie la qualité des données"""
    print("\n🔍 Vérification de la qualité des données...")
    
    # Valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"⚠️  Valeurs manquantes détectées:")
        print(missing[missing > 0])
    else:
        print("✓ Aucune valeur manquante")
    
    # Duplicats
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  {duplicates} duplicats détectés")
    else:
        print("✓ Aucun duplicat")
    
    # Types de données
    print(f"\n📊 Types de données:")
    print(df.dtypes.value_counts())
    
    return missing, duplicates

def remove_unnecessary_columns(df):
    """Supprime les colonnes inutiles"""
    print(f"\n🗑️  Suppression des colonnes: {config.FEATURES_TO_DROP}")
    df_clean = df.drop(columns=config.FEATURES_TO_DROP, errors='ignore')
    print(f"✓ {len(df_clean.columns)} colonnes conservées")
    return df_clean

def handle_outliers(df, method='iqr', threshold=3):
    """Détecte et gère les outliers"""
    print(f"\n📉 Détection des outliers (méthode: {method})...")
    
    df_clean = df.copy()
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    outliers_count = {}
    
    for col in numerical_cols:
        if col == config.TARGET_COL:
            continue
            
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                outliers_count[col] = outliers
        
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = (z_scores > threshold).sum()
            if outliers > 0:
                outliers_count[col] = outliers
    
    if outliers_count:
        print("⚠️  Outliers détectés:")
        for col, count in sorted(outliers_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {col}: {count} ({count/len(df_clean)*100:.2f}%)")
    else:
        print("✓ Aucun outlier significatif détecté")
    
    return df_clean

def encode_categorical_features(df):
    """Encode les variables catégorielles"""
    print(f"\n🏷️  Encodage des variables catégorielles...")
    
    df_encoded = df.copy()
    
    # Gender: Male=1, Female=0
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender'] = (df_encoded['Gender'] == 'Male').astype(int)
        print("✓ Gender encodé (Male=1, Female=0)")
    
    # Geography: One-Hot Encoding
    if 'Geography' in df_encoded.columns:
        geo_dummies = pd.get_dummies(df_encoded['Geography'], prefix='Geography', drop_first=True)
        df_encoded = pd.concat([df_encoded, geo_dummies], axis=1)
        df_encoded.drop('Geography', axis=1, inplace=True)
        print(f"✓ Geography encodé (One-Hot): {list(geo_dummies.columns)}")
    
    return df_encoded

def feature_engineering(df):
    """Crée de nouvelles features (Version Robuste)"""
    print(f"\n🔧 Feature Engineering (Robuste)...")
    
    df_feat = df.copy()
    
    # 1. Ratio Balance/Salary (Très pertinent, on garde)
    df_feat['BalanceSalaryRatio'] = df_feat['Balance'] / (df_feat['EstimatedSalary'] + 1)
    
    # 2. Interaction simple (au lieu de créer des groupes complexes)
    df_feat['InactiveWithProducts'] = (df_feat['NumOfProducts'] > 1) & (df_feat['IsActiveMember'] == 0)
    df_feat['InactiveWithProducts'] = df_feat['InactiveWithProducts'].astype(int)
    
    # 3. Zero Balance Flag (On garde)
    df_feat['HasZeroBalance'] = (df_feat['Balance'] == 0).astype(int)
    
    # On laisse le modèle gérer la non-linéarité tout seul.    
    print(f"✓ {len([c for c in df_feat.columns if c not in df.columns])} nouvelles features créées")
    
    # Tri des colonnes pour stabilité (Crucial pour la prod)
    df_feat = df_feat.sort_index(axis=1)
    
    return df_feat

def clean_data(df):
    """Pipeline complet de nettoyage"""
    print("\n" + "="*70)
    print("🧹 NETTOYAGE DES DONNÉES")
    print("="*70)
    
    # 1. Qualité
    check_data_quality(df)
    
    # 2. Suppression colonnes inutiles
    df_clean = remove_unnecessary_columns(df)
    
    # 3. Gestion outliers
    df_clean = handle_outliers(df_clean, method='iqr')
    
    # 4. Encodage
    df_clean = encode_categorical_features(df_clean)
    
    # 5. Feature engineering
    df_clean = feature_engineering(df_clean)
    
    print(f"\n✅ Nettoyage terminé!")
    print(f"   Dimensions finales: {df_clean.shape}")
    
    return df_clean

def save_processed_data(df, filepath=None):
    """Sauvegarde les données nettoyées"""
    if filepath is None:
        filepath = config.PROCESSED_DATA_FILE
    
    print(f"\n💾 Sauvegarde des données nettoyées vers {filepath}")
    df.to_csv(filepath, index=False)
    print("✓ Sauvegarde terminée")

def main():
    """Fonction principale"""
    # Charger
    df = load_raw_data()
    
    # Nettoyer
    df_clean = clean_data(df)
    
    # Sauvegarder
    save_processed_data(df_clean)
    
    print("\n" + "="*70)
    print("✨ Pipeline de nettoyage complété avec succès!")
    print("="*70)

if __name__ == "__main__":
    main()
