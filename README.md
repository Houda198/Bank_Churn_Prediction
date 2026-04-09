# 🏦 Bank Customer Churn Prediction

Projet de Machine Learning pour prédire le taux de désabonnement (churn) des clients bancaires.

## 📋 Description

Système de prédiction permettant d'identifier les clients à risque de quitter la banque, facilitant ainsi la mise en place d'actions de rétention ciblées.

**Cas d'usage :** Départements marketing et relation client des banques.

## 🎯 Objectifs

- Prédire les clients susceptibles de quitter la banque (churn)
- Identifier les facteurs clés de désabonnement
- Fournir des insights actionnables via dashboard Power BI
- Optimiser les campagnes de rétention

## 📊 Dataset

**Source :** [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

**Caractéristiques :**
- ~10,000 clients
- 14 features (démographie, produits, comportement)
- Target binaire : Exited (0 = actif, 1 = churné)

## 🏗️ Structure du projet
src/ : Scripts modulaires (Nettoyage, EDA, Entraînement, Évaluation).

config.py : Centralisation des paramètres (chemins, hyperparamètres, seuils).

models/ : Stockage des modèles sérialisés (.pkl) et du scaler.

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Setup

```bash
# Cloner le repo
git clone https://github.com/votre-username/bank-churn-prediction.git
cd bank-churn-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Utilisation

### 1. Télécharger le dataset
Téléchargement [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) et placement `Churn_Modelling.csv` dans `data/raw/`

### 2. Lancer le pipeline complet

```bash
python main.py
```

Cela exécute :
- ✅ Nettoyage des données
- ✅ Analyse exploratoire (graphiques sauvegardés)
- ✅ Feature engineering
- ✅ Entraînement de 4 modèles (LR, RF, XGBoost, LightGBM)
- ✅ Évaluation & comparaison
- ✅ Sauvegarde du meilleur modèle


## 📈 Résultats

Modèle,         Accuracy, Precision, Recall (Churn), F1-Score, AUC-ROC
Random Forest,  81.1%,       52.7%,       72.5%,       0.610,    0.856
XGBoost,        71.4%,       40.5%,       86.7%,       0.552,    0.867
LightGBM,       70.3%,       39.5%,       86.5%,       0.542,    0.869
Logistic Regression, 59.3%,  31.7%,       86.2%,       0.463,    0.785

## 🔑 Features importantes

1. **Age** - Le facteur n°1 (les clients matures ont des comportements de churn très marqués)
2. **NumOfProducts** - 1 seul produit = risque élevé
3. **IsActiveMember** - L'absence d'activité récente est le signal d'alerte immédiat.
4. **Geography** - Un taux de churn anormalement plus élevé sur ce segment géographique.


## 🛠️ Technologies

- **Python 3.10**
- **ML :** scikit-learn, XGBoost, LightGBM
- **Data :** pandas, numpy
- **Viz :** matplotlib, seaborn, plotly

## 📝 Méthodologie

1. **Exploration :** Analyse univariée/bivariée, détection outliers
2. **Preprocessing :** Encodage catégorielles, normalisation, gestion déséquilibre (SMOTE)
3. **Modeling :** 4 algorithmes testés avec validation croisée
4. **Tuning :** GridSearchCV pour hyperparamètres optimaux
5. **Évaluation :** Focus sur recall (détecter un maximum de churn)

## 🚀 Améliorations Clés  : 

 Contrairement à une approche standard, ce pipeline intègre :

- Threshold Tuning (0.35) : Ajustement du seuil de décision pour capturer 86% des churneurs.

- Gestion du déséquilibre : Utilisation de class_weight='balanced' et scale_pos_weight pour compenser la minorité de clients churnés (~20%).

- Feature Engineering Robuste : Création de ratios métier (Balance/Salary) et de flags d'inactivité.

## 🏆 Choix Stratégique : LightGBM

 Bien que le Random Forest ait un meilleur F1-score, nous recommandons LightGBM pour la production :

 Recall de 86.5% : On ne rate que 13% des clients qui partent.

 AUC de 0.869 : Excellente capacité du modèle à distinguer un client fidèle d'un client à risque.

## 📖 Comment reproduire les résultats : 

- Installation : pip install -r requirements.txt

- Pipeline complet : python main.py

- Évaluation avec seuil personnalisé :

- Le seuil est configurable dans evaluate.py.

- Lancement : $env:PYTHONPATH = "."; python src/evaluate.py