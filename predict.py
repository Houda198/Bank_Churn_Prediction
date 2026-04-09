"""
Script de prédiction avec modèles sauvegardés
Exemple d'utilisation des modèles entraînés
"""

import pandas as pd
import joblib
import config

def load_best_model():
    """Charge le meilleur modèle (basé sur évaluation)"""
    # Changez ici le nom du modèle si nécessaire
    best_model_name = 'XGBoost'  # ou 'RandomForest', 'LightGBM', etc.
    
    model_path = config.MODELS_DIR / f"{best_model_name}.pkl"
    scaler_path = config.MODELS_DIR / "scaler.pkl"
    
    print(f"📂 Chargement du modèle: {best_model_name}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, best_model_name

def predict_churn(customer_data, model, scaler):
    """
    Prédit le churn pour un ou plusieurs clients
    
    Args:
        customer_data: DataFrame avec les features nécessaires
        model: Modèle ML entraîné
        scaler: StandardScaler fité
        
    Returns:
        DataFrame avec prédictions et probabilités
    """
    # Normaliser
    customer_data_scaled = scaler.transform(customer_data)
    customer_data_scaled = pd.DataFrame(customer_data_scaled, 
                                        columns=customer_data.columns,
                                        index=customer_data.index)
    
    # Prédire
    predictions = model.predict(customer_data_scaled)
    probabilities = model.predict_proba(customer_data_scaled)[:, 1]
    
    # Résultats
    results = pd.DataFrame({
        'Prediction': ['Churn' if p == 1 else 'Active' for p in predictions],
        'Churn_Probability': probabilities,
        'Risk_Level': ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.4 else 'LOW' 
                       for p in probabilities]
    }, index=customer_data.index)
    
    return results

def example_prediction():
    """Exemple de prédiction sur clients fictifs"""
    
    # Charger le modèle
    model, scaler, model_name = load_best_model()
    
    # Créer des exemples de clients (features après feature engineering)
    # Note: En prod, vous devriez avoir le pipeline complet de preprocessing
    
    example_customers = pd.DataFrame({
        # Features principales
        'CreditScore': [650, 800, 450],
        'Age': [45, 28, 62],
        'Tenure': [2, 8, 1],
        'Balance': [0, 125000, 85000],
        'NumOfProducts': [1, 2, 1],
        'HasCrCard': [1, 1, 0],
        'IsActiveMember': [0, 1, 0],
        'EstimatedSalary': [50000, 120000, 30000],
        
        # Features géographiques (one-hot encodées)
        'Geography_Germany': [0, 1, 0],
        'Geography_Spain': [0, 0, 1],
        
        # Gender (1=Male, 0=Female)
        'Gender': [1, 0, 1],
        
        # Features engineerées (exemple simplifié)
        'BalanceSalaryRatio': [0, 1.04, 2.83],
        'AgeGroup_Middle': [1, 0, 0],
        'AgeGroup_Senior': [0, 0, 1],
        'AgeGroup_Elder': [0, 0, 0],
        'TenureGroup_Medium': [0, 1, 0],
        'TenureGroup_Long': [0, 0, 0],
        'ProductsActivity': [0, 2, 0],
        'HasZeroBalance': [1, 0, 0]
    }, index=['Client_A', 'Client_B', 'Client_C'])
    
    # Prédire
    print(f"\n🔮 Prédiction avec {model_name}\n")
    results = predict_churn(example_customers, model, scaler)
    
    # Afficher
    print("="*60)
    print("RÉSULTATS DES PRÉDICTIONS")
    print("="*60)
    print(results)
    print("\n")
    
    # Détails par client
    for client_id in results.index:
        prob = results.loc[client_id, 'Churn_Probability']
        pred = results.loc[client_id, 'Prediction']
        risk = results.loc[client_id, 'Risk_Level']
        
        print(f"📊 {client_id}:")
        print(f"   Statut prédit: {pred}")
        print(f"   Probabilité churn: {prob*100:.1f}%")
        print(f"   Niveau de risque: {risk}")
        print()
    
    return results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 EXEMPLE DE PRÉDICTION DE CHURN")
    print("="*60)
    
    try:
        results = example_prediction()
        
        print("\n✅ Prédictions réussies!")
        print("\n💡 Prochaines étapes:")
        print("   1. Utilisez ce script comme template")
        print("   2. Adaptez pour vos données de production")
        print("   3. Intégrez dans votre API/dashboard")
        
    except FileNotFoundError:
        print("\n❌ Modèles non trouvés!")
        print("   Exécutez d'abord: python main.py")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
