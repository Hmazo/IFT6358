#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os
import pickle 
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

# Login to Wandb
wandb.login()

# Save the model and metrics
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        
def plot_roc_curve(fpr, tpr, auc_score):
    sns.set()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plt.plot(fpr, tpr, label='DecisionTree (AUC = {:.3f})'.format(auc_score), lw=3)
    plt.plot([0, 1], [0, 1], 'k--', label='Ligne de base aléatoire (AUC = 0.500)', lw=3)
    ax.set_ylabel('Taux de vrais positifs', fontsize=18)
    ax.set_xlabel('Taux de faux positifs', fontsize=18)
    ax.set_title("Courbes ROC", fontsize=18)
    ax.tick_params(labelsize=12)
    plt.legend(loc='best')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig("roc_curve.png")
    wandb.log({"ROC Curve": wandb.Image("roc_curve.png")})
    plt.show()
    
def percentage_formatter(x, pos):
    return f'{x * 100:.0f}%'

def plot_taux_reussite_but(df_prob=None, n_bins=20, quant=5, list_labels=None):
    palette_couleurs = ['darkorange', 'green', 'navy', 'red']
    compteur_but, compteur_tir, goal_rate, pencentil = [], [], [], []

    for col in df_prob.columns[1:-1]:
        df_prob['percentile'] = df_prob[col].rank(pct=True)
        quantile_list = np.linspace(0, 1, n_bins * quant + 1).round(4).tolist()
        q = df_prob.quantile(quantile_list)
        for i in np.arange(quant, (quant * n_bins) + 1, quant):
            df_percentil = df_prob[(df_prob[col] >= q[col][(i - quant) / 100]) & (df_prob[col] < q[col][i / 100])]
            compteur_but.append(df_percentil['IsGoal'].sum())
            compteur_tir.append(df_percentil['compteur_tir'].sum())
            goal_rate.append(df_percentil['IsGoal'].sum() / df_percentil['compteur_tir'].sum() if df_percentil['compteur_tir'].sum() > 0 else 0)
            pencentil.append(i)

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    for i in range(len(list_labels)):
        sns.lineplot(x=pencentil[i * n_bins:n_bins * (i + 1) - 1],
                     y=goal_rate[i * n_bins:n_bins * (i + 1) - 1],
                     label=f'{list_labels[i]}', color=palette_couleurs[i], legend=False, linewidth=3)
    ax.set_xlim(left=105, right=-5)
    ax.set_ylim(bottom=0, top=1)
    ax.legend(fontsize=12)
    ax.set_ylabel('But / (Tir + But)')
    ax.set_xlabel('Percentile du modèle de probabilité de tir')
    ax.set_title(f"Taux de réussite des buts vs. Percentile du modèle de probabilité de tir")
    ax.set_xticks(np.arange(0, 101, 10))
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.grid(color='white', linestyle='--', linewidth=0.8)
    plt.savefig("goal_success_rate.png")
    wandb.log({"Goal Success Rate": wandb.Image("goal_success_rate.png")})
    plt.show()

def plot_taux_reussite_cumulatif(df_prob=None, n_bins=20, quant=5, list_labels=None):
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    palette_couleurs = ['darkorange', 'green', 'navy', 'red', 'magenta']
    compteur_but, pencentil, cum_goal_rate = [], [], []
    df_prob = df_prob.copy()

    for col in df_prob.columns[1:-1]:
        temp, cumulative_goals = 0, 0
        df_prob['percentile'] = df_prob[col].rank(pct=True)
        quantile_list = np.linspace(0, 1, n_bins * quant + 1).round(4).tolist()
        q = df_prob.quantile(quantile_list)
        total_goals = df_prob['IsGoal'].sum()

        for j in np.arange((quant * n_bins), 0, -quant):
            df_perc = df_prob[(df_prob[col] > q[col][(j - quant) / 100]) & (df_prob[col] <= q[col][j / 100])]
            compteur_but.append(df_perc.IsGoal.sum())
            cumulative_goals += df_perc.IsGoal.sum()
            cum_goal_rate.append(cumulative_goals / total_goals)
            pencentil.append(j)

    for i in range(len(list_labels)):
        sns.lineplot(x=pencentil[i * n_bins:n_bins * (i + 1) - 1], y=cum_goal_rate[i * n_bins:n_bins * (i + 1) - 1],
                     label=f'{list_labels[i]}', color=palette_couleurs[i], legend=False, linewidth=3)

    ax.set_xlim(left=105, right=-5)
    ax.set_ylim(bottom=0, top=max(cum_goal_rate) + 0.05)
    ax.set_ylabel('Proportion de buts cumulés')
    ax.set_xlabel('Percentile du modèle de probabilité de tir')
    ax.set_title(f"Taux de réussite cumulatif des buts vs. Percentile du modèle de probabilité de tir")
    ax.legend(fontsize=12)
    plt.grid(color='white', linestyle='--', linewidth=0.8)
    ax.set_xticks(np.arange(0, 101, 10))
    plt.savefig("cumulative_goal_rate.png")
    wandb.log({"Cumulative Goal Rate": wandb.Image("cumulative_goal_rate.png")})
    plt.show()

def main():
    # Initialize Wandb
    wandb.init(
        project="IFT6758.2024-A",
        name="Decision_Tree_Q6",
        notes="Decison Tree with hyperparameter tunning via RandomizedSearchCV",
        config={
            "test_size": 0.30,
            "random_state": 0,
            "n_iter": 10,
            "cv": 5,
            "scoring": "accuracy",
            "model": "DecisionTreeClassifier",
        }
    )
    config = wandb.config

    # Load data
    #file_path = os.path.join('dataframe', 'train_data_enriched.csv')
    df_enriched = pd.read_csv('dataframe/train_data_enriched.csv')

    data_categ = df_enriched.select_dtypes(include=['object']).drop(['attacking_team_name', 'home_team'], axis=1)
    data_numer = df_enriched.select_dtypes(include=['float64', 'int64']).drop(
        ['game_id', 'game_seconds', 'game_period', 'time_since_last_event', 'attacking_team_id'], axis=1
    )
    data_bool = df_enriched.select_dtypes(include=['bool'])
    
    le = LabelEncoder()
    data_categ = data_categ.apply(le.fit_transform)
    df_final = pd.concat([data_categ, data_numer, data_bool], axis=1).dropna()
    
    X = df_final.drop(['is_goal'], axis=1)
    y = df_final['is_goal']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)

    # RandomizedSearchCV hyperparam tunning
    param_dist = {
        'splitter': ['best', 'random'],
        'max_depth': stats.randint(1, 30),
        'min_samples_split': stats.randint(2, 11),
        'min_samples_leaf': stats.randint(1, 9),
        'max_features': stats.uniform(0, 1),
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    
    rs = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(),
        param_distributions=param_dist,
        n_iter=config.n_iter,
        cv=config.cv,
        scoring=config.scoring,
        random_state=config.random_state
    )
    rs.fit(X_train, y_train)
    best_params = rs.best_params_
    wandb.config.update(best_params)  # Log best hyperparameters

    # Trainning
    optimized_tree = DecisionTreeClassifier(**best_params)
    decision_tree = optimized_tree.fit(X_train, y_train)
    y_pred = optimized_tree.predict(X_valid)
    probs = optimized_tree.predict_proba(X_valid)[:, 1]

    # metrics
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)
    roc_auc = roc_auc_score(y_valid, probs)


    


    # Predictions
    fpr, tpr, thresholds = roc_curve(y_valid, probs)
    auc_score = auc(fpr, tpr)

    # Log plots
    #plot_roc_curve(fpr, tpr, auc_score)

    df_prob = pd.DataFrame({'IsGoal': y_valid, 'proba': probs})
    df_prob['compteur_tir'] = 1
    #plot_taux_reussite_but(df_prob=df_prob, list_labels=['Decision Tree Model'])
    #plot_taux_reussite_cumulatif(df_prob=df_prob, list_labels=['Decision Tree Model'])
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    #wandb.log(metrics)  # Log metrics to Wandb
    


    metrics_data = {
    "Model": ["Decision Tree"],
    "Accuracy": [accuracy_score(y_valid, y_pred)],
    "Precision": [precision_score(y_valid, y_pred)], 
    "Recall": [recall_score(y_valid, y_pred)],        
    "F1 Score": [f1_score(y_valid, y_pred)],          
    "ROC AUC": [roc_auc_score(y_valid, probs)]  
    }

    # Create a DataFrame for metrics
    metrics_table = pd.DataFrame(metrics_data)

    # Log the metrics table to Wandb
    wandb.log({"Metrics Table": wandb.Table(dataframe=metrics_table)})


    # Save model and predictions
    save_model(decision_tree, 'model_decision_tree.pkl')
    wandb.save('model_decision_tree.pkl')  # Upload model file to Wandb
    
    # Save predictions and probabilities
    np.save('y_pred_tree.npy', y_pred)
    np.save('probs_tree.npy', probs)
    np.save('y_valid_tree.npy', y_valid)

    wandb.log({
        "predictions": wandb.Histogram(y_pred),
        "probabilities": wandb.Histogram(probs),
        "true_labels": wandb.Histogram(y_valid)
    })
    
    prediction_table = pd.DataFrame({
    "True Labels": y_valid,
    "Predicted Labels": y_pred,
    "Predicted Probabilities": probs
    })

    wandb.log({"Predictions Table": wandb.Table(dataframe=prediction_table)})

    wandb.finish()

if __name__ == "__main__":
    main()


# In[ ]:




