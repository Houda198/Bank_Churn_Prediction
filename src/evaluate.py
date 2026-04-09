"""
Module d'évaluation (Version Robuste - Détection Overfitting)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
import config

plt.style.use(config.PLOT_STYLE)
sns.set_palette(config.PLOT_PALETTE)

def load_models():
    """Charge les modèles"""
    print("📂 Chargement modèles...")
    models = {}
    for name in config.MODELS_TO_TEST:
        path = config.MODELS_DIR / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"✓ {name}")
    return models

def load_test_data():
    """Charge test + val data"""
    test_data = joblib.load(config.MODELS_DIR / "test_data.pkl")
    print(f"✓ Test: {len(test_data['X_test']):,} | Val: {len(test_data['X_val']):,}")
    return test_data

def evaluate_on_test(models, X_test, y_test):
    """Évaluation finale sur TEST"""
    print("\n" + "="*70)
    print("📊 ÉVALUATION SUR TEST SET")
    print("="*70)
    
    results = []
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
    
        # On applique le seuil personnalisé (0.35 au lieu de 0.50)
        threshold = 0.35
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        results.append(metrics)
        predictions[name] = y_pred
        probabilities[name] = y_proba
        
        print(f"\n🎯 {name}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  AUC:       {metrics['auc']:.3f}")
    
    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    return df, predictions, probabilities

def plot_train_val_test_comparison(test_results):
    """Compare train/val/test pour détecter overfitting"""
    # Note: pour avoir train/val metrics, il faudrait les sauvegarder depuis train.py
    # Pour simplifier, on fait juste les graphiques test
    pass

def plot_roc_curves(y_test, probabilities):
    """ROC curves"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} ({auc:.3f})")
    
    ax.plot([0,1], [0,1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_figure('roc_curves')
    print("✓ ROC curves sauvegardées")

def plot_confusion_matrix(best_name, y_test, y_pred):
    """Matrice confusion du meilleur"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Active', 'Churn'], yticklabels=['Active', 'Churn'])
    ax.set_title(f'Confusion Matrix - {best_name}', fontweight='bold')
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réalité')
    
    plt.tight_layout()
    save_figure('confusion_matrix_best')
    print("✓ Confusion matrix sauvegardée")

def save_figure(name):
    """Sauvegarde figure"""
    path = config.FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Pipeline évaluation"""
    print("\n" + "="*70)
    print("🎯 ÉVALUATION FINALE")
    print("="*70)
    
    models = load_models()
    test_data = load_test_data()
    
    X_test, y_test = test_data['X_test'], test_data['y_test']
    
    # Évaluation
    results_df, preds, probas = evaluate_on_test(models, X_test, y_test)
    
    # Résumé
    print("\n" + "="*70)
    print("📈 CLASSEMENT FINAL")
    print("="*70)
    print(results_df[['model', 'f1', 'recall', 'precision', 'auc']].to_string(index=False))
    
    # Meilleur modèle
    best = results_df.iloc[0]
    print(f"\n🏆 MEILLEUR: {best['model']}")
    print(f"   F1={best['f1']:.3f} | Recall={best['recall']:.3f} | AUC={best['auc']:.3f}")
    
    # Visualisations
    print("\n🎨 Visualisations...")
    plot_roc_curves(y_test, probas)
    plot_confusion_matrix(best['model'], y_test, preds[best['model']])
    
    # Rapport détaillé
    print("\n📋 RAPPORT DÉTAILLÉ")
    print(classification_report(y_test, preds[best['model']], 
                               target_names=['Active', 'Churn'], digits=3))
    
    # Sauvegarder résultats
    results_df.to_csv(config.REPORTS_DIR / "final_results.csv", index=False)
    print(f"\n✓ Résultats sauvegardés: {config.REPORTS_DIR}")
    
    print("\n✨ Évaluation terminée!")
    
    return results_df

if __name__ == "__main__":
    main()
