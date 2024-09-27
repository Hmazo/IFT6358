import requests
import json
import os

def get_data(year_list):
    """
    Fetches and saves NHL season data for a given range of years.
    
    Args:
        year_list (list): A list containing the start and end years (inclusive) 
                     for which data is to be fetched.
                     
    The function iterates over each year and each game type (regular season and playoffs),
    checks if the game data already exists in the local JSON file, and only downloads 
    new data if not present. After fetching the data, it saves the updated season data 
    into a JSON file.
    """
    start, end = year_list[0], year_list[1]

    game_type_map = {
        "02": "regular_season",
        "03": "playoffs",
    }
 

    for year_i in range(start, end):
        season_ids =[]
        season_data = {  
            "data": [],
        }
        print(f"Getting data for season {year_i}")
        file_name = f"Data/nhl_season_{year_i}.json"
        if os.path.exists(file_name):
            # Load existing season data if the file exists
            print(f"File data for season {year_i} exist. Loading Data")
            with open(file_name, 'r') as file:
                season_data = json.load(file)                   
                for game in season_data["data"]:
                    id = game.get("id",None)
                    season_ids.append(id)
            print(f"File data for season {year_i} is loaded.")
            
        for game_type,_ in game_type_map.items():         
            for i in range(1, 10000):
                game_id = str(year_i) + game_type + f'{i:04}'
                
                # Check if the game ID already exists in the file
                if int(game_id) in season_ids:
                    print(f"Game ID {game_id} already exists. Skipping download.")
                    continue
                print(f"loading data for id {game_id}")
                url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
                response = request(url, game_id)
            
                if response is not None:
                    season_data["data"].append(response)
                else:
                    break  # Stop fetching if no data is found for the game

        # Save the data after processing all game types
        save_data(year_i, season_data)



def request(url, id):
    """
    Sends a request to the NHL API to retrieve play-by-play data for a specific game.
    
    Args:
        url (str): The API endpoint URL for the game data.
        id (str): The unique identifier of the game.
        
    Returns:
        dict: The JSON response if the request is successful.
        None: If the request fails or the game data is not found.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {id}. Status code: {response.status_code}")
        return None

def save_data(year, data):
    """
    Saves the fetched data for a given season into a JSON file.
    
    Args:
        year (int): The year of the season being saved.
        data (dict): The dictionary containing the season data.
    
    Saves the data in a JSON file named `nhl_season_<year>.json`.
    """
    file_name = f"Data/nhl_season_{year}.json"
    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data for season {year} saved successfully.")
    except Exception as e:
        print(f"Error saving data for season {year}: {e}")

# get data from season 2016-2017 to season 2023-2024
year = [2016, 2024]
get_data(year)
