"""
Script principal - Pipeline complet de prédiction de churn
"""

import sys
from pathlib import Path
import config

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import data_cleaning, eda, train, evaluate

def print_banner(text):
    """Affiche un banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_full_pipeline(use_grid_search=False):
    """Exécute le pipeline complet"""
    
    print_banner("🏦 BANK CUSTOMER CHURN PREDICTION - PIPELINE COMPLET")
    
    # Vérifier que le dataset existe
    if not config.RAW_DATA_FILE.exists():
        print(f"\n❌ ERREUR: Dataset non trouvé!")
        print(f"   Attendu: {config.RAW_DATA_FILE}")
        print(f"\n💡 Téléchargez 'Churn_Modelling.csv' depuis Kaggle et placez-le dans data/raw/")
        return
    
    try:
        # ÉTAPE 1: Nettoyage des données
        print_banner("ÉTAPE 1/4: NETTOYAGE DES DONNÉES")
        df_raw = data_cleaning.load_raw_data()
        df_clean = data_cleaning.clean_data(df_raw)
        data_cleaning.save_processed_data(df_clean)
        
        # ÉTAPE 2: Analyse exploratoire
        print_banner("ÉTAPE 2/4: ANALYSE EXPLORATOIRE (EDA)")
        eda.generate_eda_report(df_raw)  # Sur raw pour voir Geography/Gender
        
        # ÉTAPE 3: Entraînement
        print_banner("ÉTAPE 3/4: ENTRAÎNEMENT DES MODÈLES")
        trained_models, X_test, y_test, train_results = train.main(use_grid_search=use_grid_search)   
             
        # ÉTAPE 4: Évaluation
        print_banner("ÉTAPE 4/4: ÉVALUATION DES MODÈLES")
        results_df, best_model = evaluate.main()
        
        # RÉSUMÉ FINAL
        print_banner("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
        print(f"\n🏆 Meilleur modèle: {best_model}")
        print(f"\n📊 Résultats complets:")
        print(results_df[['model_name', 'f1_score', 'recall', 'precision', 'roc_auc']].to_string(index=False))
        
        print(f"\n📁 Fichiers générés:")
        print(f"   - Données nettoyées: {config.PROCESSED_DATA_FILE}")
        print(f"   - Modèles: {config.MODELS_DIR}")
        print(f"   - Graphiques: {config.FIGURES_DIR}")
        print(f"   - Rapports: {config.REPORTS_DIR}")
        
        print("\n🚀 Projet prêt pour portfolio!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERREUR lors du pipeline: {e}")
        import traceback
        traceback.print_exc()

def run_step(step_name):
    """Exécute une étape spécifique"""
    
    steps = {
        'clean': ('Nettoyage des données', data_cleaning.main),
        'eda': ('Analyse exploratoire', eda.main),
        'train': ('Entraînement des modèles', lambda: train.main(use_grid_search=False)),
        'evaluate': ('Évaluation des modèles', evaluate.main)
    }
    
    if step_name not in steps:
        print(f"❌ Étape '{step_name}' non reconnue")
        print(f"Étapes disponibles: {list(steps.keys())}")
        return
    
    step_title, step_function = steps[step_name]
    print_banner(f"🎯 {step_title.upper()}")
    
    try:
        step_function()
        print(f"\n✅ {step_title} terminée!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline de prédiction de churn bancaire")
    parser.add_argument('--step', type=str, help='Exécuter une étape spécifique (clean/eda/train/evaluate)')
    parser.add_argument('--grid-search', action='store_true', help='Utiliser GridSearch pour tuning (plus lent)')
    
    args = parser.parse_args()
    
    if args.step:
        run_step(args.step)
    else:
        run_full_pipeline(use_grid_search=args.grid_search)
