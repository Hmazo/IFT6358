#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
import os

# Ensure parent directory is in Python's module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from serving_client import ServingClient
from game_client import GameClient

# Initialize the serving client with the correct IP and port (use port 5000 if that's where your Flask app runs)
serving_client = ServingClient(ip='localhost', port=5001)  # Changed to port 5000 assuming Flask runs here

st.title("NHL Game Analysis")

# Model configuration and download (sidebar)
with st.sidebar:
    st.header("Configure Model")
    workspace = st.text_input("Enter Workspace", "a10-ift6758-milestone-2")
    model_name = st.text_input("Enter Model Name", "model_distance")
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
        processed_data = game_client.process_events(game_data)
        if not processed_data.empty:
            st.write("Processed Game Data:", processed_data)
            # Make predictions
            predictions = game_client.make_predictions(processed_data)
            if not predictions.empty:
                st.subheader("Predictions")
                st.dataframe(predictions)
            else:
                st.error("No predictions available. Check processed data and prediction setup.")
        else:
            st.error("No data to process.")
    else:
        st.error("Failed to fetch game data.")

# Logs and/or additional game info
if st.button("Show Logs"):
    logs = serving_client.logs()
    if 'error' not in logs:
        st.write(logs)
    else:
        st.error(f"Failed to retrieve logs: {logs['error']}")
