import streamlit as st
import pandas as pd
import requests
import wandb

# Initialisation de l'application Streamlit
st.title("Tableau de bord interactif pour xG Predictions")

# Section 1 : Téléchargement du modèle depuis WandB
st.header("Télécharger le modèle depuis WandB")

workspace = st.text_input("Workspace", value="IFT6758.2024-A")
model_name = st.text_input("Model Name", value="expected_goals_model")
version = st.text_input("Model Version", value="v1")

if st.button("Télécharger le modèle"):
    with st.spinner("Téléchargement en cours depuis WandB..."):
        # Télécharger le modèle via l'API W&B
        api = wandb.Api()
        model_artifact = api.artifact(f"{workspace}/{model_name}:{version}")
        model_path = model_artifact.download()
        st.success(f"Modèle téléchargé dans {model_path}")

# Section 2 : Requête pour un jeu spécifique
st.header("Requête de données de jeu")

game_id = st.text_input("Identifiant de jeu", value="2021020329")

if st.button("Requête Jeu"):
    with st.spinner("Récupération des données du jeu..."):
        # Ping l'API du client de jeu pour obtenir les données
        game_data = requests.get(f"http://127.0.0.1:<PORT>/game/{game_id}").json()
        
        # Extraire les informations nécessaires
        home_team = game_data["home_team"]
        away_team = game_data["away_team"]
        period = game_data["period"]
        time_remaining = game_data["time_remaining"]
        score_home = game_data["score_home"]
        score_away = game_data["score_away"]
        xg_home = sum(game_data["xG_home"])
        xg_away = sum(game_data["xG_away"])
        diff_home = xg_home - score_home
        diff_away = xg_away - score_away
        
        # Affichage des résultats
        st.write(f"**Équipes** : {home_team} vs {away_team}")
        st.write(f"**Période** : {period}")
        st.write(f"**Temps restant** : {time_remaining}")
        st.write(f"**Score actuel** : {home_team} {score_home} - {score_away} {away_team}")
        st.write(f"**Somme des xG** : {home_team} {xg_home:.2f} - {xg_away:.2f} {away_team}")
        st.write(f"**Différence xG/score** : {home_team} {diff_home:.2f} - {diff_away:.2f} {away_team}")

# Section 3 : Afficher le tableau des événements avec les prédictions
st.header("Tableau des événements de tir et prédictions")

if st.button("Afficher les prédictions"):
    with st.spinner("Chargement des prédictions du modèle..."):
        # Requête au service de prédiction
        events_data = requests.get(f"http://127.0.0.1:<PORT>/predict", json={"game_id": game_id}).json()
        events_df = pd.DataFrame(events_data)
        
        # Affichage du dataframe
        st.dataframe(events_df)
