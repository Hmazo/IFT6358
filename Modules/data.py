import requests
import json
import os

def get_data(year_list):
    """
    Fetches and saves NHL season data for a given range of years.
    
    Args:
        year_list (list): A list containing the start and end years (inclusive) 
                     for which data is to be fetched.
    """
    start, end = year_list[0], year_list[1]

    game_type_map = {
        "02": "regular_season",
        "03": "playoffs",
    }
 
    for year_i in range(start, end):
        
        for game_type,_ in game_type_map.items():
            for i in range(1, 1300):
                game_id = str(year_i) + game_type + f'{i:04}'
                # Create filename based on season, year, and game ID
                file_name = f"Data/nhl_season_{year_i}_{game_id}.json"
                
                # Check if the game file already exists
                if os.path.exists(file_name):
                    continue
                url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
                response = request(url, game_id)

                if response is not None:
                    save_game_data(file_name, response)
                else:
                    continue
                    


def request(url, game_id):
    """
    Sends a request to the NHL API to retrieve play-by-play data for a specific game.
    
    Args:
        url (str): The API endpoint URL for the game data.
        game_id (str): The unique identifier of the game.
        
    Returns:
        dict: The JSON response if the request is successful.
        None: If the request fails or the game data is not found.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for game ID {game_id}. Status code: {response.status_code}")
        return None

def save_game_data(file_name, data):
    """
    Saves the fetched game data into a JSON file.
    
    Args:
        file_name (str): The name of the JSON file to save the game data.
        data (dict): The dictionary containing the game data.
    """
    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data for game saved successfully in {file_name}.")
    except Exception as e:
        print(f"Error saving game data in {file_name}: {e}")


def load_data(year_list):
    """
    Loads NHL game data for the given years in chronological order.
    
    Args:
        year_list (list): A list containing the years to load data from, in chronological order.
        
    Returns:
        list: A list of game data loaded from the JSON files, ordered from oldest to most recent.
    """
    all_data={}
    for year in year_list:
        print(f"Loading data for season {year}")
        
        # Get all files for the season year that match the format nhl_season_year_*.json
        files = [f for f in os.listdir("Data") if f.startswith(f"nhl_season_{year}_") and f.endswith(".json")]
        
        files.sort()
        temp =[]
        # Load data from each file and append to all_data
        for file_name in files:
            file_path = os.path.join("Data", file_name)
            
            with open(file_path, 'r') as f:
                game_data = json.load(f)
                temp.append(game_data)
                
        all_data[year]=temp
        
        print(f"Data for season {year} loaded successfully.")
    
    return all_data