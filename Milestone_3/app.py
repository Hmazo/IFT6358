#!/usr/bin/env python
# coding: utf-8

"""
If you are in the same directory as this file (app.py), you can run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn
"""

import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import pickle
import wandb
import sys



app = Flask(__name__)

# Ensure WANDB_API_KEY is set
#################################export WANDB_API_KEY=your_api_key ####Run this in your terminal !!!!!!###############

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
if WANDB_API_KEY is None:
    app.logger.error("La variable d'environnement WANDB_API_KEY n'est pas définie.")
    raise ValueError("La variable d'environnement WANDB_API_KEY n'est pas définie.")
wandb.login(key=WANDB_API_KEY)

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

def download_model(project, model_name, version):
    """Download a model from WandB and load it."""
    try:
        artifact = wandb.use_artifact(f'{project}/{model_name}:{version}', type='model')
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            app.logger.error(f"Model file {model_name}.pkl not found in artifact directory: {artifact_dir}")
            return None

        clf = pickle.load(open(model_path, 'rb'))
        app.logger.info(f'Le modèle {model_name} (alias {version}) a été téléchargé avec succès depuis WandB.')
        return clf
    except Exception as e:
        app.logger.error(f'Erreur lors du téléchargement du modèle {model_name} (alias {version}): {e}')
        return None

def get_carac(clf_name):
    if "distance_model" in clf_name:
        return ['shot_distance']
    elif "angle_model" in clf_name:
        return ['shot_angle']
    elif "combined_model" in clf_name:
        return ['shot_distance', 'shot_angle']
    return None

def rename_model_file(clf_name):
    """Rename model files for consistency if needed (not currently used)."""
    pass

with app.app_context():
    """Initialization before the first request."""
    # Setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    logging.info('Lancement du serveur')

    # Clear the log file
    with open(LOG_FILE, 'w'):
        pass

    # Initialize WandB
    wandb.init(project="IFT6758.2024-A", entity="hicham-mazouzi-university-of-montreal", name="flask_server")
    app.logger.info("WandB initialized successfully.")

    # Download default models if not already downloaded
    global clf_name
    global clf

    # Here, we use ':latest' alias because we dont have a version linked to each model
    # Adding other models
    json_models = [
        {'project': 'hicham-mazouzi-university-of-montreal/IFT6758.2024-A', 'model_name': 'combined_model', 'version': 'latest'}
    ]

    # Download the default (combined_model:latest)
    for m in json_models:
        clf = download_model(m['project'], m['model_name'], m['version'])
        if clf is not None:
            clf_name = f"{m['model_name']}_{m['version']}" # Set clf_name according to model_name and version
        else:
            app.logger.warning(f"Could not load model {m['model_name']}:{m['version']}")

@app.route("/", methods=["GET"])
def index():
    return "Test"

@app.route("/checking", methods=["GET"])
def health():
    return {'message': 'Tout est correct!'}

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response."""
    with open(LOG_FILE) as f:
        response = f.readlines()
    return jsonify(response)

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """Handles POST requests to download a specific model from WandB."""
    payload = request.get_json()
    app.logger.info(payload)

    global clf_name, clf

    new_clf_name = f'{payload["clf"]}_{payload["version"]}'
    try:
        project = "hicham-mazouzi-university-of-montreal/IFT6758.2024-A"
        clf = download_model(project, payload['clf'], payload['version'])
        if clf is None:
            raise ValueError(f"Unable to load {payload['clf']}:{payload['version']}")
        clf_name = new_clf_name
        wandb.log({"model_downloaded": new_clf_name})
        return jsonify({"model_name": new_clf_name, "status": "Téléchargé avec succès."})
    except Exception as e:
        app.logger.error(f'Erreur lors du téléchargement du modèle {new_clf_name}: {e}')
        return jsonify({"error": "Erreur lors du téléchargement du modèle."})

@app.route("/predict", methods=["POST"])
def predict():
    """Handles POST requests to make predictions using the loaded model."""
    payload = request.get_json()
    app.logger.info(payload)

    global clf_name

    app.logger.info(f'Le model qui sera utilisé est {clf_name}')

    try:
        if clf_name is not None:
            carac = get_carac(clf_name)
            if carac is None:
                app.logger.warning("Les caractéristiques ne sont pas définies pour le clf_name actuel.")
                return jsonify({"error": "Les caractéristiques ne sont pas définies pour le clf_name actuel."})

            X = pd.DataFrame(payload)[carac]

            y_pred = clf.predict_proba(X)[:, 1]

            response = y_pred.tolist()
            wandb.log({"prediction_success": True, "predictions": response})
            
            return jsonify(response)

        else:
            app.logger.warning("clf_name n'est pas défini.")
            return jsonify({"error": "clf_name n'est pas défini."})

    except Exception as e:
        app.logger.warning(f'Les prédictions n\'ont pas pu être calculées: {e}')
        wandb.log({"prediction_success": False})
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
