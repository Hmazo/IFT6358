import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay

def generate_roc_auc_data( y_val,y_pred_proba):
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}


def generate_goal_rate_data(y_val, y_pred_proba):
    percentiles = np.percentile(y_pred_proba, np.arange(0, 101, 1))
    goal_rates = []
    for i in range(len(percentiles) - 1):
        mask = (y_pred_proba >= percentiles[i]) & (y_pred_proba < percentiles[i + 1])
        if mask.sum() > 0:
            goal_rates.append(np.mean(y_val[mask]))
        else:
            goal_rates.append(0)
    return np.arange(100, 0, -1), goal_rates  # Reverse percentiles for the x-axis


def generate_cumulative_goal_data(y_val, y_pred_proba):
    # Reset index to align with positional indices
    y_val_reset = y_val.reset_index(drop=True)
    
    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_y = y_val_reset.iloc[sorted_indices]
    
    # Compute cumulative goals
    cumulative_goals = np.cumsum(sorted_y) / np.sum(sorted_y) * 100  # Convert to percentage
    
    # Match cumulative goals to 100 percentiles
    percentiles = np.linspace(0, 100, len(cumulative_goals))
    cumulative_goals_percentiles = np.interp(np.arange(100, 0, -1), percentiles, cumulative_goals)
    
    return np.arange(100, 0, -1), cumulative_goals_percentiles



def generate_calibration_data(y_val, y_pred_proba):
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10, strategy='uniform')
    return prob_pred, prob_true

def plot_roc_auc(roc_data, model_labels, filename=""):
    plt.figure(figsize=(12, 8))
    sns.set()
    for i, data in enumerate(roc_data):
        plt.plot(data['fpr'], data['tpr'], label=f'{model_labels[i]} (AUC = {data["roc_auc"]:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Ligne de base aléatoire')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_goal_rate(goal_rate_data, model_labels, filename=""):
    plt.figure(figsize=(12, 8))
    sns.set()
    for i, (x, y) in enumerate(goal_rate_data):
        plt.plot(x, y, label=f'{model_labels[i]}', lw=2)
    plt.xlabel('Percentile du modèle de probabilité de tir')
    plt.ylabel('But / (Tir + But)')
    plt.title('Taux de réussite des buts vs. Percentile du modèle de probabilité de tir')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_cumulative_goals(cumulative_data, model_labels, filename=""):
    plt.figure(figsize=(12, 8))
    sns.set()
    for i, (x, y) in enumerate(cumulative_data):
        plt.plot(x, y, label=f'{model_labels[i]}', lw=2)
    plt.xlabel('Percentile du modèle de probabilité de tir')
    plt.ylabel('Proportion de buts cumulés')
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
    y_axis = [0,10,20,30,40,50,60,70,80,90,100]
    y_values = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
    plt.yticks(y_axis, y_values)
    plt.title('Taux de réussite cumulatif des buts vs. Percentile du modèle de probabilité de tir')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_calibration(y_val_list, y_pred_list, model_labels, filename=""):
    plt.figure(figsize=(12, 8))
    sns.set()
    for i, (y_val, y_pred) in enumerate(zip(y_val_list, y_pred_list)):
        CalibrationDisplay.from_predictions(y_val, y_pred, n_bins=10, label=model_labels[i])
    plt.title('Tracés de calibration (courbe de fiabilité)')
    plt.xlabel('Valeur moyenne prédite"')
    plt.ylabel('Fraction de positifs')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

