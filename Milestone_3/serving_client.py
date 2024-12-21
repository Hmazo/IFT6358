import json
import requests
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)
APP = os.environ.get("APP", "0.0.0.0") 

logger = logging.getLogger(__name__)
APP = os.environ.get("APP", "0.0.0.0")  

class ServingClient:
    def __init__(self, ip='localhost', port=5001):  # Assuming Flask runs on port 5001(else: change to 5000 !)
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initialized ServingClient with base URL: {self.base_url}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sends input data to the prediction service and retrieves predictions."""
        try:
            # chacking that X contain the correct columns expected by our Flask server
            if 'shot_distance' not in X.columns or 'shot_angle' not in X.columns:
                return pd.DataFrame({"error": ["DataFrame must contain 'shot_distance' and 'shot_angle' columns"]})
            
            # Convert DataFrame to JSON 
            json_data = X.to_json(orient='records')
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{self.base_url}/predict",
                data=json_data,
                headers=headers,
                timeout=10  # timeout for the request
            )
            response.raise_for_status()  # flag an exception for bad HTTP status codes

            # Convert JSON response to df
            predictions = response.json()
            if isinstance(predictions, list):  # Checking if the response is a list as expected
                return pd.DataFrame(predictions, columns=['prediction'])
            else:
                return pd.DataFrame({"error": ["Invalid response format from server"]})

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return pd.DataFrame({"error": [str(e)]})
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return pd.DataFrame({"error": [str(e)]})

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
            # Post request with model details for download !
            r = requests.post(
                f"{self.base_url}/download_registry_model",
                json={"workspace": workspace, "clf": clf, "version": version},
                timeout=10  # Set timeout to avoid indefinite waiting
            )
            r.raise_for_status()  # Raise HTTP Error for bad responses !

            logger.info(f"Model {clf}-{version} successfully downloaded")
            return r.json()  

        except requests.HTTPError as http_err:
            # Log and handle HTTP-specific errors
            logger.error(f"HTTP error during model download {clf}-{version}: {http_err}")
            return {"error": f"HTTP error: {http_err}"}
        except Exception as e:
            # Log and handle generic errors
            logger.error(f"Error during model download {clf}-{version}: {e}")
            return {"error": f"Failed to download model: {e}"}
        
        
        
    def logs(self):
        """ Retrieves logs from the Flask server. """
        try:
            response = requests.get(f"{self.base_url}/logs")
            response.raise_for_status()  # Ensuring that any HTTP error is flagged ! 
            return response.json()  
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve logs: {e}")
            return {"error": str(e)}
        except ValueError as e:
            logger.error("Failed to parse JSON response from logs.")
            return {"error": str(e)}