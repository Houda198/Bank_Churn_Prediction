"""
Module d'analyse exploratoire des données (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config

# Configuration plots
plt.style.use(config.PLOT_STYLE)
sns.set_palette(config.PLOT_PALETTE)

def load_data(filepath=None):
    """Charge les données"""
    if filepath is None:
        # Essayer processed d'abord, sinon raw
        if config.PROCESSED_DATA_FILE.exists():
            filepath = config.PROCESSED_DATA_FILE
        else:
            filepath = config.RAW_DATA_FILE
    
    print(f"📂 Chargement: {filepath}")
    df = pd.read_csv(filepath)
    return df

def overview_analysis(df):
    """Analyse globale du dataset"""
    print("\n" + "="*70)
    print("📊 VUE D'ENSEMBLE")
    print("="*70)
    
    print(f"\nDimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"\nAperçu des colonnes:")
    print(df.dtypes)
    
    print(f"\nStatistiques descriptives:")
    print(df.describe())
    
    # Distribution de la target
    if config.TARGET_COL in df.columns:
        churn_rate = df[config.TARGET_COL].mean()
        print(f"\n🎯 Taux de churn: {churn_rate*100:.2f}%")
        print(f"   - Churned: {df[config.TARGET_COL].sum():,}")
        print(f"   - Active: {(df[config.TARGET_COL]==0).sum():,}")

def plot_target_distribution(df):
    """Visualise la distribution de la target"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    target_counts = df[config.TARGET_COL].value_counts()
    axes[0].bar(['Active', 'Churned'], target_counts.values, 
                color=[config.COLORS['secondary'], config.COLORS['danger']])
    axes[0].set_ylabel('Nombre de clients')
    axes[0].set_title('Distribution de la Target')
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(target_counts.values, labels=['Active', 'Churned'], autopct='%1.1f%%',
                colors=[config.COLORS['secondary'], config.COLORS['danger']])
    axes[1].set_title('Proportion Active vs Churned')
    
    plt.tight_layout()
    save_figure('01_target_distribution')
    print("✓ Graphique target sauvegardé")

def plot_numerical_distributions(df):
    """Distribution des variables numériques"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [c for c in numerical_cols if c != config.TARGET_COL][:6]  # Top 6
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        df[col].hist(bins=30, ax=axes[i], color=config.COLORS['primary'], alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution: {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Fréquence')
    
    plt.tight_layout()
    save_figure('02_numerical_distributions')
    print("✓ Distributions numériques sauvegardées")

def plot_churn_by_category(df):
    """Taux de churn par catégories"""
    categorical_analysis = []
    
    # Geography (si existe dans raw data)
    if 'Geography' in df.columns:
        categorical_analysis.append('Geography')
    
    # Gender (si existe)
    if 'Gender' in df.columns:
        categorical_analysis.append('Gender')
    
    if not categorical_analysis:
        print("⚠️  Pas de variables catégorielles à analyser")
        return
    
    fig, axes = plt.subplots(1, len(categorical_analysis), figsize=(12, 4))
    if len(categorical_analysis) == 1:
        axes = [axes]
    
    for i, cat in enumerate(categorical_analysis):
        churn_rate = df.groupby(cat)[config.TARGET_COL].mean() * 100
        churn_rate.plot(kind='bar', ax=axes[i], color=config.COLORS['danger'], alpha=0.7)
        axes[i].set_title(f'Taux de Churn par {cat}')
        axes[i].set_ylabel('Taux de Churn (%)')
        axes[i].set_xlabel(cat)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs
        for j, v in enumerate(churn_rate.values):
            axes[i].text(j, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    save_figure('03_churn_by_category')
    print("✓ Churn par catégorie sauvegardé")

def plot_correlation_matrix(df):
    """Matrice de corrélation"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Matrice de Corrélation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure('04_correlation_matrix')
    print("✓ Matrice de corrélation sauvegardée")

def plot_churn_by_numerical(df):
    """Distributions numériques par statut churn"""
    key_features = ['Age', 'Balance', 'NumOfProducts', 'CreditScore']
    key_features = [f for f in key_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(key_features):
        df[df[config.TARGET_COL]==0][feature].hist(bins=30, alpha=0.6, 
                                                    label='Active', 
                                                    color=config.COLORS['secondary'], 
                                                    ax=axes[i])
        df[df[config.TARGET_COL]==1][feature].hist(bins=30, alpha=0.6, 
                                                    label='Churned', 
                                                    color=config.COLORS['danger'], 
                                                    ax=axes[i])
        axes[i].set_title(f'{feature} par statut Churn')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Fréquence')
        axes[i].legend()
    
    plt.tight_layout()
    save_figure('05_churn_by_numerical')
    print("✓ Variables numériques par churn sauvegardées")

def save_figure(filename):
    """Sauvegarde une figure"""
    filepath = config.FIGURES_DIR / f"{filename}.{config.FIGURE_FORMAT}"
    plt.savefig(filepath, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()

def generate_eda_report(df):
    """Génère un rapport EDA complet"""
    print("\n" + "="*70)
    print("📈 ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("="*70)
    
    # Overview
    overview_analysis(df)
    
    # Visualisations
    print("\n🎨 Génération des visualisations...")
    plot_target_distribution(df)
    plot_numerical_distributions(df)
    plot_churn_by_category(df)
    plot_correlation_matrix(df)
    plot_churn_by_numerical(df)
    
    print(f"\n✅ EDA terminée! Graphiques dans {config.FIGURES_DIR}")

def main():
    """Fonction principale"""
    df = load_data()
    generate_eda_report(df)
    
    print("\n" + "="*70)
    print("✨ EDA complétée avec succès!")
    print("="*70)

if __name__ == "__main__":
    main()
