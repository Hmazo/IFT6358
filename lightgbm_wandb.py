#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
import pickle
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import scipy.stats as stats
import lightgbm as lgb
import warnings
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import pickle
import wandb

# Initialize Wandb login
wandb.login()

# Save the model and metrics as Wandb artifacts
def save_model_artifact(model, model_name):
    model_file = f"{model_name}.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(model_file)
    wandb.log_artifact(artifact)

def save_metrics_artifact(metrics, model_name):
    metrics_file = f"{model_name}_metrics.txt"
    with open(metrics_file, 'w') as file:
        file.write(str(metrics))
    artifact = wandb.Artifact(f"{model_name}_metrics", type="metrics")
    artifact.add_file(metrics_file)
    wandb.log_artifact(artifact)

def main():
    # Initialize Wandb
    wandb.init(
        project="IFT6758.2024-A",
        name="lightgbm_Q6",
        config={
            "test_size": 0.30,
            "random_state": 0,
            "model": "LightGBM",
            "num_boost_round": 100,
        }
    )
    config = wandb.config

    # Load data
    df_enriched= pd.read_csv('dataframe/train_data_enriched.csv')

    # Preprocess data
    data_categ = df_enriched.select_dtypes(include=['object']).drop(['attacking_team_name', 'home_team'], axis=1)
    data_numer = df_enriched.select_dtypes(include=['float64', 'int64']).drop(
        ['game_id', 'game_seconds', 'game_period', 'time_since_last_event', 'attacking_team_id'], axis=1
    )
    data_bool = df_enriched.select_dtypes(include=['bool'])

    le = LabelEncoder()
    data_categ = data_categ.apply(le.fit_transform)

    df_final = pd.concat([data_categ, data_numer, data_bool], axis=1)
    X = df_final.drop(['is_goal'], axis=1)
    y = df_final['is_goal']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)

    # Train LightGBM model
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=50,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary',
        metric='auc',
        random_state=config.random_state
    )
    lightgbm = clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    #y_prediction = clf.predict(X_valid)
    #proba = clf.predict_proba(X_valid)[:, 1]
    
    
    
    #### features selection:
    lgb.plot_importance(clf, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
    gain_selection= clf.feature_importances_

    gain_selection_df = pd.DataFrame({
    'Features': X_train.columns,
    'Importance_Gain': gain_selection,
    })

    # Sort df by importance 
    features_df = gain_selection_df.sort_values(by='Importance_Gain', ascending=False)
    features = features_df['Features'].head(20).tolist()
    data_selected = df_final[features]
    X_selected = data_selected
    y_selected = df_final['is_goal']

    X_train, X_valid, y_train, y_valid = train_test_split(X_selected, y_selected, test_size=0.30, random_state=0)

    clf = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               metric='auc', min_child_samples=20, min_child_weight=0.001,
               min_split_gain=0.1, n_estimators=100, n_jobs=-1, num_leaves=40,
               objective='binary', random_state=None, reg_alpha=0.0,
               reg_lambda=0.0, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0, num_boost_round=100)

    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    y_predictions=clf.predict(X_valid)
    proba = clf.predict_proba(X_valid)[:, 1]
    
    
    
    
    # Calculate metrics
    accuracy = accuracy_score(y_valid, y_predictions)
    auc_score = roc_auc_score(y_valid, proba)
    metrics = {
        "accuracy": accuracy,
        "roc_auc": auc_score,
        "precision": precision_score(y_valid, y_predictions),
        "recall": recall_score(y_valid, y_predictions),
        "f1_score": f1_score(y_valid, y_predictions),
    }
    print(f"Metrics: {metrics}")
    wandb.log(metrics)  # Log metrics to Wandb
    
    metrics_data = {
    "Model": ["lightgbm"],
    "Accuracy": [accuracy_score(y_valid, y_predictions)],
    "Precision": [precision_score(y_valid, y_predictions)], 
    "Recall": [recall_score(y_valid, y_predictions)],        
    "F1 Score": [f1_score(y_valid, y_predictions)],          
    "ROC AUC": [roc_auc_score(y_valid, proba)]  
    }

    # Create a DataFrame for metrics
    metrics_table = pd.DataFrame(metrics_data)

    # Log the metrics table to Wandb
    wandb.log({"Metrics Table": wandb.Table(dataframe=metrics_table)})

    # Save and log model artifact
    save_model_artifact(lightgbm, "lightgbm_model")

    # Save and log metrics artifact
    save_metrics_artifact(metrics, "lightgbm_model")

    # Log predictions and probabilities as a table
    preds_table = pd.DataFrame({
        "y_true": y_valid,
        "y_pred": y_predictions,
        "proba": proba
    })
    wandb.log({"Predictions Table": wandb.Table(dataframe=preds_table)})

    # Optional: Log classification report as text
    report = classification_report(y_valid, y_predictions)
    wandb.log({"Classification Report": report})

    # Save predictions and probabilities locally
    np.save('y_pred_lightgbm.npy', y_predictions)
    np.save('probs_lightgbm.npy', proba)
    np.save('y_valid_lightgbm.npy', y_valid)

    # Finish Wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

