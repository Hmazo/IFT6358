import sys
import os

# Ensure parent directory is in Python's module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import streamlit as st
from serving_client import ServingClient
from game_client import GameClient

# Initialize the serving client
serving_client = ServingClient(ip='localhost', port=5001)

st.title("NHL Game Analysis")

# Sidebar for model configuration
with st.sidebar:
    st.header("Configure Model")
    workspace = st.text_input("Enter Workspace", "a10-ift6758-milestone-3")
    model_name = st.selectbox("Enter Model Name", ["choose a model", "distance_model", "angle_model", "combined_model"])
    model_version = st.text_input("Enter Model Version", "latest")
    if st.button("Download Model"):
        response = serving_client.download_registry_model(workspace, model_name, model_version)
        if 'error' not in response:
            st.success("Model downloaded successfully!")
        else:
            st.error(f"Failed to download model: {response['error']}")

# Game ID and fetching game data
game_id = st.text_input("Enter Game ID", "2021020329")
if st.button("Fetch Game Data"):
    game_client = GameClient(game_id, serving_client)
    game_data = game_client.fetch_events()

    if game_data:
        st.success("Game data fetched successfully!")
        
        # Extract game details for display
        home_team = f"{game_data['homeTeam']['placeName']['default']} {game_data['homeTeam']['commonName']['default']}"
        away_team = f"{game_data['awayTeam']['placeName']['default']} {game_data['awayTeam']['commonName']['default']}"
        current_period = game_data.get('periodDescriptor', {}).get('number', 'Unknown')
        time_remaining = game_data.get('clock', {}).get('timeRemaining', 'Unknown')
        home_score = game_data['homeTeam']['score']
        away_score = game_data['awayTeam']['score']

        # Process game events
        processed_data = game_client.process_events(game_data)

        # Pass processed data through make_predictions
        predicted_data = game_client.make_predictions(processed_data)

        if not predicted_data.empty:
            if 'prediction' in predicted_data.columns:
                # Calculate xG for both teams
                xg_home = predicted_data[predicted_data['attacking_team_name'] == home_team]['prediction'].sum()
                xg_away = predicted_data[predicted_data['attacking_team_name'] == away_team]['prediction'].sum()

                # Calculate differences between actual score and xG
                xg_diff_home = round(xg_home - home_score, 2)
                xg_diff_away = round(xg_away - away_score, 2)

                # Display middle section
                st.subheader(f"Game {game_id}: {away_team} vs {home_team}")
                st.markdown(f"**Period:** {current_period} | **Time Remaining:** {time_remaining}")
                st.markdown(f"**Score:** {away_team} {away_score} - {home_team} {home_score}")
                st.markdown(f"**Expected Goals (xG):** {away_team} {xg_away:.1f} ({xg_diff_away:+.1f}) - {home_team} {xg_home:.1f} ({xg_diff_home:+.1f})")

                # Show processed data and predictions
                st.write("Data Used For Predictions (And Predictions):", predicted_data)
            else:
                st.error("Predictions could not be made. Ensure the prediction service is running and the processed data is valid.")
        else:
            st.error("No data available for predictions.")
    else:
        st.error("Failed to fetch game data.")

# Logs
if st.button("Show Logs"):
    logs = serving_client.logs()
    if 'error' not in logs:
        st.write(logs)
    else:
        st.error(f"Failed to retrieve logs: {logs['error']}")
