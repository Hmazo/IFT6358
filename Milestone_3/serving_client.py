import json
import requests
import pandas as pd
import logging
import os


logger = logging.getLogger(__name__)
APP = os.environ.get("APP", "0.0.0.0")  # Default to localhost if APP environment variable is not set


class ServingClient:
    def __init__(self, ip: str = APP, port: int = 5000, features=None):
        """
        Initializes the ServingClient.

        Args:
            ip (str): The IP address or hostname of the server.
            port (int): The port number for the server.
            features (list): The list of features expected by the prediction model.
        """
        self.base_url = f"http://{ip}:{port}"  # Base URL for API requests
        logger.info(f"Initializing client; base URL: {self.base_url}")

        # Set default features if none are provided
        if features is None:
            features = ["distance"]
        self.features = features

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Sends input data to the prediction service and retrieves predictions.

        Args:
            X (pd.DataFrame): Input DataFrame with the features required by the model.

        Returns:
            pd.DataFrame: A DataFrame containing predictions or an error message.
        """
        logger.info("Initializing prediction request...")

        # Validate that the input DataFrame contains all required features
        if not all(feature in X.columns for feature in self.features):
            raise ValueError(f"Input DataFrame must contain the following columns: {self.features}")

        try:
            # Send POST request with input data serialized as JSON
            r = requests.post(
                f"{self.base_url}/predict",
                json=json.loads(X.to_json()),  # Convert DataFrame to JSON format
                timeout=10  # Set timeout to avoid indefinite waiting
            )
            r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Convert server response to a DataFrame
            predictions = r.json()
            logger.info("Predictions successfully generated")
            return pd.DataFrame({"prediction": predictions}, index=X.index)

        except requests.HTTPError as http_err:
            # Log and handle HTTP-specific errors
            logger.error(f"HTTP error during prediction: {http_err}")
            return pd.DataFrame({"error": [str(http_err)]})
        except Exception as e:
            # Log and handle generic errors
            logger.error(f"Error during prediction: {e}")
            return pd.DataFrame({"error": [str(e)]})

    def logs(self) -> dict:
        """
        Retrieves the server logs.

        Returns:
            dict: A dictionary containing the server logs or an error message.
        """
        logger.info("Initializing request to retrieve logs...")

        try:
            # Send GET request to retrieve logs
            r = requests.get(f"{self.base_url}/logs", timeout=10)
            r.raise_for_status()  # Raise HTTPError for bad responses

            logger.info("Logs successfully retrieved")
            return r.json()  # Return logs as JSON

        except requests.HTTPError as http_err:
            # Log and handle HTTP-specific errors
            logger.error(f"HTTP error retrieving logs: {http_err}")
            return {"error": f"HTTP error: {http_err}"}
        except Exception as e:
            # Log and handle generic errors
            logger.error(f"Error retrieving logs: {e}")
            return {"error": f"Failed to retrieve logs: {e}"}

    def download_registry_model(self, workspace: str, clf: str, version: str) -> dict:
        """
        Requests the service to download and switch to a specific model version.

        Args:
            workspace (str): The WandB workspace containing the model.
            clf (str): The name of the model in the registry.
            version (str): The version of the model to download.

        Returns:
            dict: A dictionary containing the result of the model download request.
        """
        logger.info(f"Initializing request to download model {clf}-{version}...")

        try:
            # Send POST request with model details for download
            r = requests.post(
                f"{self.base_url}/download_registry_model",
                json={"workspace": workspace, "clf": clf, "version": version},
                timeout=10  # Set timeout to avoid indefinite waiting
            )
            r.raise_for_status()  # Raise HTTPError for bad responses

            logger.info(f"Model {clf}-{version} successfully downloaded")
            return r.json()  # Return server response as JSON

        except requests.HTTPError as http_err:
            # Log and handle HTTP-specific errors
            logger.error(f"HTTP error during model download {clf}-{version}: {http_err}")
            return {"error": f"HTTP error: {http_err}"}
        except Exception as e:
            # Log and handle generic errors
            logger.error(f"Error during model download {clf}-{version}: {e}")
            return {"error": f"Failed to download model: {e}"}
