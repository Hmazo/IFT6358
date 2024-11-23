#!/usr/bin/env python
# coding: utf-8

# In[6]:

from Modules import Ingenierie, data
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Initialize Wandb login
wandb.login()

def load_data():
    file_path = os.path.join('dataframe', 'combined_shots_data.csv')
    return pd.read_csv(file_path)

# Save the model as a Wandb artifact
def save_model_artifact(model, model_name):
    model_file = f"{model_name}.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(model_file)
    wandb.log_artifact(artifact)

# Save the metrics to Wandb
def save_metrics_artifact(metrics, model_name):
    metrics_file = f"{model_name}_metrics.txt"
    with open(metrics_file, 'w') as file:
        file.write(str(metrics))
    artifact = wandb.Artifact(f"{model_name}_metrics", type="metrics")
    artifact.add_file(metrics_file)
    wandb.log_artifact(artifact)

def random_baseline(X_train, y_train, X_valid, y_valid, name):
    # Generate random predictions from a uniform distribution
    random_predictions = np.random.uniform(0, 1, len(X_valid))

    # Convert the random predictions to binary (0 or 1) based on a threshold
    threshold = 0.5
    y_pred = (random_predictions >= threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Model {name} Accuracy: {accuracy:.2f}")

    return y_pred, random_predictions

def regression_logistique(X_train, y_train, X_valid, y_valid, model_name, model_type):
    # Dictionnaires pour les courbes ROC
    taux_faux_positif = dict()
    taux_vrai_positif = dict()
    aire_sous_courbe = dict()

    # Listes pour stocker les données
    liste_taux_faux_positif = []
    liste_taux_vrai_positif = []
    liste_aire_sous_courbe = []
    liste_caractéristiques = []

    # Listes pour les statistiques de buts et de tirs
    pourcentage_buts = []
    percentile = []
    cumul_but = []

    # List pour stocker les tags
    tags = []

    # -------- Modèle --------
    if model_type == 'logistic_regression':
        # Create an instance of Logistic Regression Classifier and fit the data.
        model = LogisticRegression().fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        predicted_prob = model.predict_proba(X_valid)
    elif model_type == 'random_baseline':
        model = None
        # Your custom random baseline model here
        y_pred, predicted_prob = random_baseline(X_train, y_train, X_valid, y_valid, model_name)
    else:
        raise ValueError("Invalid model_type. Use 'logistic_regression' or 'random_baseline'.")

    acc = accuracy_score(y_valid, y_pred)
    print("Le score de précision de l'entraînement d'un classificateur de régression logistique en fonction de {} est : {}".format(model_name, acc))
    print("Les probabilités prédites sont : ")
    print(predicted_prob)

    accuracy = accuracy_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "recall": recall,
    }
    params = {
            "model_type": 'Regression logistique sur {}'.format(model_name),
            "scaler": "standard scaler",
            "param_grid": str(model.get_params()) if model_type == 'logistic_regression' else "N/A",
    }


    # -------- ROC --------
    if len(predicted_prob.shape) > 1 and predicted_prob.shape[1] > 1:
        p = predicted_prob[:, 1]
    else:
        p = predicted_prob
        
    taux_faux_positif, taux_vrai_positif, _ = roc_curve(y_valid, p)
    aire_sous_courbe = auc(taux_faux_positif, taux_vrai_positif)
    liste_taux_vrai_positif.append(taux_vrai_positif)
    liste_taux_faux_positif.append(taux_faux_positif)
    liste_aire_sous_courbe.append(aire_sous_courbe)
    liste_caractéristiques.append(model_name)

    # -------- Goal Rate --------
    new_series = np.array(y_valid)
    new_series = np.reshape(new_series, (new_series.shape[0]))

    # Scale true_probabilities (predict_proba() returns true and false both) on percentile scale (0-100)
    true_prob = pd.DataFrame()
    true_prob['true_target'] = np.array(new_series)
    percentile = [[np.percentile(p, i), np.percentile(p, i+5)] 
                  for i in range(0,100,5)]

    # Looping on probabilities to check their percentiles with their status as goal/shot
    for i in range(len(percentile)):
        true_prob_percentile = true_prob[(p<=percentile[i][1]) & (p>percentile[i][0])]
        goals = true_prob_percentile.loc[true_prob_percentile['true_target']==1].shape[0]
        shots = true_prob_percentile.loc[true_prob_percentile['true_target']==0].shape[0]
        
        if goals == 0:
            pourcentage_buts.append(0)
        else:
            pourcentage_buts.append((goals*100)/(goals+shots))

    shot_prob_model_percentile = np.arange(0, 100, 5)

    # -------- Cumulative --------
    new_series = np.array(y_valid)
    new_series = np.reshape(new_series, (new_series.shape[0]))

    # Scale true_probabilities (predict_proba() returns true and false both) on percentile
    true_prob = pd.DataFrame()
    true_prob['true_target'] = np.array(new_series)
    percentile = [[np.percentile(p, i), np.percentile(p, i+1)] 
                  for i in range(0,100,1)]
    total_goal = np.sum(new_series)

    # Looping on probabilities to check their percentiles with their status as goal/shot
    for i in range(0, len(percentile)-1):
        # We need previous and current goal lie in the percentile
        true_prob_percentile = true_prob[(p>=percentile[i][0])]
        goals = true_prob_percentile.loc[true_prob_percentile['true_target']==1].shape[0]
        # If no goal, do nothing, calculate the formula if goal
        cumul_but.append(goals*100/total_goal)
    cumul_but.append(0)

    # Axis for percentile
    shot_prob_model_percentile2 = np.arange(0, 100, 1)

    return model, taux_faux_positif, taux_vrai_positif, aire_sous_courbe, shot_prob_model_percentile, pourcentage_buts, y_pred, shot_prob_model_percentile2, cumul_but, p, metrics

def main():
    # Initialize Wandb
    wandb.init(project="IFT6758.2024-A",name="modeles_de_bases_Q3",notes="3 modeles logistiques+random baseline",config={"random_state": 42, "test_size": 0.3})
    
    df = load_data()
    
    df_distance_goal = df[["shot_distance", "is_goal"]].dropna()
    df_angle_goal = df[["shot_angle", "is_goal"]].dropna()
    df_angle_distance_goal = df[["shot_distance", "shot_angle", "is_goal"]].dropna()
    
    features_distance = df_distance_goal[['shot_distance']]
    features_angle = df_angle_goal[['shot_angle']]
    features_combined = df_angle_distance_goal[['shot_distance', 'shot_angle']]
    target_distance = df_distance_goal['is_goal']
    target_angle = df_angle_goal['is_goal']
    target_combined = df_angle_distance_goal['is_goal']
    X_random = df_distance_goal[['shot_distance']]
    y_random = df_distance_goal['is_goal']

    # Data split
    X_train_distance, X_valid_distance, y_train_distance, y_valid_distance = train_test_split(features_distance, target_distance, test_size=0.3, random_state=42)
    X_train_angle, X_valid_angle, y_train_angle, y_valid_angle = train_test_split(features_angle, target_angle, test_size=0.3, random_state=42)
    X_train_combined, X_valid_combined, y_train_combined, y_valid_combined = train_test_split(features_combined, target_combined, test_size=0.3, random_state=42)
    X_train_random, X_valid_random, y_train_random, y_valid_random = train_test_split(X_random, y_random, test_size=0.3, random_state=42)
    
    models = [
        ('distance', X_train_distance, y_train_distance, X_valid_distance, y_valid_distance, 'logistic_regression'),
        ('angle', X_train_angle, y_train_angle, X_valid_angle, y_valid_angle, 'logistic_regression'),
        ('combined', X_train_combined, y_train_combined, X_valid_combined, y_valid_combined, 'logistic_regression'),
        ('random', X_train_distance, y_train_distance, X_valid_distance, y_valid_distance, 'random_baseline')
    ]

    for model_name, X_train, y_train, X_valid, y_valid, model_type in models:
        (model,taux_faux_positif, taux_vrai_positif, aire_sous_courbe, 
         shot_prob_model_percentile, pourcentage_buts, y_pred, 
         shot_prob_model_percentile2, cumul_but, p, metrics) = regression_logistique(X_train, y_train, X_valid, y_valid, model_name, model_type)
        
        print(f"Metrics for {model_name}: {metrics}")
        
        # Log metrics to Wandb
        wandb.log({
            f"{model_name}_accuracy": metrics["accuracy"],
            f"{model_name}_precision": metrics["precision"],
            f"{model_name}_recall": metrics["recall"],
            f"{model_name}_f1_score": metrics["f1"],
        })
        
        # Save and log model artifact
        if model_type != 'random_baseline'and model is not None:
            save_model_artifact(model, f"{model_name}_model")
            

        
        # Save and log metrics artifact
        save_metrics_artifact(metrics, model_name)
        
    metrics_list = []

    for model_name, X_train, y_train, X_valid, y_valid, model_type in models:
        (taux_faux_positif, taux_vrai_positif, aire_sous_courbe, 
         shot_prob_model_percentile, pourcentage_buts, y_pred, 
         shot_prob_model_percentile2, cumul_but, p, metrics) = regression_logistique(X_train, y_train, X_valid, y_valid, model_name, model_type)
        
        print(f"Metrics for {model_name}: {metrics}")
        

        metrics_list.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
        })
        
        # Save and log metrics artifact
        save_metrics_artifact(metrics, model_name)
    
    # Create a DataFrame for all metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    # Log the metrics table to Wandb
    wandb.log({"Metrics Table": wandb.Table(dataframe=metrics_df)})
    
    wandb.finish()


if __name__ == "__main__":
    main()


# In[ ]:
