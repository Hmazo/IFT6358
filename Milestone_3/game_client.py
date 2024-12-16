import requests
import logging
import pandas as pd
import math
from serving_client import ServingClient


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
            # Make a GET request to the NHL API
            response = requests.get(self.api_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            game_data = response.json()  # Parse the JSON response
            logger.info(f"Fetched data for game {self.game_id}.")
            return game_data
        except Exception as e:
            # Log errors if fetching data fails
            logger.error(f"Error fetching events for game {self.game_id}: {e}")
            return {}

    def process_events(self, game_data: dict) -> pd.DataFrame:
        """
        Processes game data to extract features for prediction, including manual calculation
        of shot distance and shot angle.

        Args:
            game_data (dict): Raw game data from the NHL API.

        Returns:
            pd.DataFrame: DataFrame containing processed event features.
        """
        logger.info(f"Processing events for game {self.game_id}...")

        # List to store shot data
        shots_data = []

        for event in game_data.get('plays', []):
            # Extract the type of event
            event_type = event.get('typeDescKey')

            # Process only relevant event types
            if event_type in ["shot-on-goal", "goal", "blocked-shot", "missed-shot"]:
                details = event.get('details', {})
                x_coord = details.get('xCoord')
                y_coord = details.get('yCoord')

                # Skip events with missing coordinates
                if x_coord is None or y_coord is None:
                    logger.warning(f"Skipping event due to missing coordinates: {event}")
                    continue

                # Calculate the distance to the goal
                goal_x = 89 if x_coord > 0 else -89
                distance = math.sqrt((x_coord - goal_x) ** 2 + y_coord ** 2)

                # Calculate the angle to the goal
                if x_coord != goal_x:  # Avoid division by zero
                    angle_radians = math.atan(abs(y_coord) / abs(x_coord - goal_x))
                    angle_degrees = math.degrees(angle_radians)

                    # Adjust the angle based on the shot's quadrant
                    if y_coord < 0:
                        angle_degrees *= -1  # Reflect negative angles below the goal
                else:
                    angle_degrees = 0  # Set angle to 0 when directly aligned with the goal

                # Append processed shot data
                shots_data.append({
                    "shot_distance": distance,
                    "shot_angle": angle_degrees
                })

        # Convert the processed shot data into a DataFrame
        processed_shots_df = pd.DataFrame(shots_data)

        # Log the processed data
        logger.info(f"Processed {len(processed_shots_df)} shot events for game {self.game_id}.")
        return processed_shots_df

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
            # Send the processed events to the prediction service
            predictions = self.prediction_client.predict(processed_events)

            # Add predictions to the processed events DataFrame
            processed_events["prediction"] = predictions["prediction"]
            return processed_events
        except Exception as e:
            # Log any errors during prediction
            logger.error(f"Error during prediction: {e}")
            return pd.DataFrame()

    def run(self):
        """
        Main method to fetch, process, and predict game events.
        """
        # Step 1: Fetch game data
        game_data = self.fetch_events()

        # Step 2: Process relevant events
        processed_events = self.process_events(game_data)

        # Step 3: Send processed events for predictions
        if not processed_events.empty:
            predicted_events = self.make_predictions(processed_events)

            # Step 4: Append new predictions to the main DataFrame
            self.data = pd.concat([self.data, predicted_events], ignore_index=True)
            logger.info(f"Processed {len(predicted_events)} new events for game {self.game_id}.")
        else:
            logger.info("No events to process for this game.")





