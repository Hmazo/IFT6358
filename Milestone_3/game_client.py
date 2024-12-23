import requests
import logging
import pandas as pd
import math
import os
import sys
from serving_client import ServingClient

# Add parent directory to sys.path for importing ingenierie_2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from Modules import Ingenierie_2

logger = logging.getLogger(__name__)

class GameClient:
    def __init__(self, game_id: str, prediction_client: ServingClient):
        """
        Initializes the GameClient.

        Args:
            game_id (str): The ID of the NHL game to process.
            prediction_client (ServingClient): An instance of ServingClient to interact with the prediction service.
        """
        self.game_id = game_id
        self.prediction_client = prediction_client
        self.api_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        self.processed_events = set()  # To track processed events
        self.data = pd.DataFrame()  # DataFrame to store processed events and predictions

    def fetch_events(self) -> dict:
        """
        Fetches raw event data for the given game ID from the NHL API.

        Returns:
            dict: Raw JSON data containing game events.
        """
        logger.info(f"Fetching events for game {self.game_id}...")
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            game_data = response.json()
            logger.info(f"Fetched data for game {self.game_id}.")
            return game_data
        except Exception as e:
            logger.error(f"Error fetching events for game {self.game_id}: {e}")
            return {}

    def process_events(self, game_data: dict) -> pd.DataFrame:
        """
        Processes game data to include enriched features while retaining custom calculations for `shot_distance` and `shot_angle`.

        Args:
            game_data (dict): Raw game data from the NHL API.

        Returns:
            pd.DataFrame: DataFrame containing enriched event features with custom calculations for distance and angle.
        """
        logger.info(f"Processing events for game {self.game_id}...")

        # Extract enriched shot data using Ingenierie_2 functions
        enriched_shot_data = Ingenierie_2.extract_shots_with_previous_and_skater_info(game_data, season="2023")
        enriched_shot_data = Ingenierie_2.add_gameplay_features(enriched_shot_data)

        # Override shot_distance and shot_angle with custom calculations
        custom_shots_data = []
        for _, row in enriched_shot_data.iterrows():
            x_coord = row['x_coord']
            y_coord = row['y_coord']

            if pd.notna(x_coord) and pd.notna(y_coord):
                goal_x = 89 if x_coord > 0 else -89
                distance = math.sqrt((x_coord - goal_x) ** 2 + y_coord ** 2)

                if x_coord != goal_x:
                    angle_radians = math.atan(abs(y_coord) / abs(x_coord - goal_x))
                    angle_degrees = math.degrees(angle_radians)
                    if y_coord < 0:
                        angle_degrees *= -1
                else:
                    angle_degrees = 0

                row['shot_distance'] = distance
                row['shot_angle'] = angle_degrees

            custom_shots_data.append(row)

        final_df = pd.DataFrame(custom_shots_data)
        logger.info(f"Processed {len(final_df)} events for game {self.game_id}.")
        return final_df

    def make_predictions(self, processed_events: pd.DataFrame) -> pd.DataFrame:
        """
        Sends processed events to the prediction service and retrieves predictions.

        Args:
            processed_events (pd.DataFrame): DataFrame with features for prediction.

        Returns:
            pd.DataFrame: Processed events with predictions added.
        """
        logger.info(f"Making predictions for game {self.game_id}...")
        if processed_events.empty:
            logger.info("No new events to predict.")
            return pd.DataFrame()

        try:
            predictions = self.prediction_client.predict(processed_events)
            processed_events["prediction"] = predictions["prediction"]
            return processed_events
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return pd.DataFrame()

    def run(self):
        """
        Main method to fetch, process, and predict game events.
        """
        game_data = self.fetch_events()
        processed_events = self.process_events(game_data)
        if not processed_events.empty:
            predicted_events = self.make_predictions(processed_events)
            self.data = pd.concat([self.data, predicted_events], ignore_index=True)
            logger.info(f"Processed {len(predicted_events)} new events for game {self.game_id}.")
        else:
            logger.info("No events to process for this game.")
