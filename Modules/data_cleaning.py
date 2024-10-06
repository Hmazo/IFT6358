import pandas as pd
import requests
import os

players= {}


def request(url, id):
    """
    Sends a GET request to the specified URL and retrieves the response data.

    Args:
        url (str): The URL to request data from.
        id (str): The ID associated with the request (used for error reporting).
    
    Returns:
        dict or None: the function returns the JSON response as a dictionary. Otherwise, it returns None.
    """
    response = requests.get(url)

    if response.status_code == 200:
        return response.json() 
    else:
        return None
    
def get_player_name(id):
    """
    Retrieves the first name of the player associated with the given player ID.

    Args:
        id (str): The unique player ID used to fetch player data from the NHL API.
    
    Returns:
        str or None: The first name of the player if found, otherwise None.
    
    The function checks if the player data is already cached in the `players` dictionary. 
    If not, it fetches the player data from the NHL API and stores it in the cache.
    """
    url = f"https://api-web.nhle.com/v1/player/{id}/landing"
    response = None
    if id not in players:
        response = request(url, id)
        if response is not None: 
            players[id] = response
            response = response['firstName']['default']
    else:
        response = players[id]['firstName']['default']
 
    return response

def get_force(situation_code, event_owner_team_id, home_team_id, away_team_id):
    """
    Determines the game situation (e.g., Power Play, Shorthanded, Even Strength) 
    based on the number of skaters and goalies for both teams.

    Args:
        situation_code (str): A 4-digit string representing the game situation:
                              - 1st digit: number of away goalies.
                              - 2nd digit: number of away skaters.
                              - 3rd digit: number of home skaters.
                              - 4th digit: number of home goalies.
        event_owner_team_id (str): The team ID associated with the event being analyzed.
        home_team_id (str): The team ID for the home team.
        away_team_id (str): The team ID for the away team.
    
    Returns:
        str: A string describing the game situation (e.g., "Power Play", "Shorthanded", 
             "Even Strength", or "Empty Net Power Play"). Returns "Unknown" if the situation
             cannot be determined.
    """
    away_goalie = int(situation_code[0])
    away_skaters = int(situation_code[1])
    home_skaters = int(situation_code[2])
    home_goalie = int(situation_code[3])

    # Check if both teams have an equal number of skaters
    if away_skaters == home_skaters:
        return "Even Strength"
    elif away_skaters > home_skaters:
        if event_owner_team_id == away_team_id:
            return "Power Play" if away_goalie == 1 else "Empty Net Power Play"
        else:
            return "Shorthanded"
    elif home_skaters > away_skaters:
        if event_owner_team_id == home_team_id:
            return "Power Play" if home_goalie == 1 else "Empty Net Power Play"
        else:
            return "Shorthanded"
    else:
        return "Unknown"

def create_dataframes(year_list, data):
    """
    Crée des DataFrames pour les jeux et les sauvegarde en fichiers CSV.

    :param year_list: Un dictionnaire où la clé est l'année et la valeur est une liste d'ID de jeux.
    :param data: Une liste contenant les données de chaque jeu.
    """

    
    # Créer un dossier pour sauvegarder les DataFrames s'il n'existe pas
    output_dir = 'dataframe'
    os.makedirs(output_dir, exist_ok=True)

    for year in year_list:
        for game in data[year]:
            
            game_id = game['id']
            home_team = game['homeTeam']['name']['default']
            away_team = game['awayTeam']['name']['default']

            dfPlays = pd.DataFrame.from_records(game['plays'])
            df_filtered = dfPlays[dfPlays['typeDescKey'].isin(['goal', 'shot-on-goal'])]

            plays_data = []
            for idx, row in df_filtered.iterrows():
                event_id = row['eventId']
                sort_order = row['sortOrder']
                time_in_period = row['timeInPeriod']
                period_number = row['periodDescriptor']['number']
                situation_code = row['situationCode']
                x_coord = row['details'].get('xCoord', None)
                y_coord = row['details'].get('yCoord', None)
                shot_type = row['details'].get('shotType', None)
                shooter_id = get_player_name(row['details'].get('shootingPlayerId', row['details'].get('scoringPlayerId', None)))
                goalie_id = get_player_name(row['details'].get('goalieInNetId', None))
                event_owner_team_id = row['details']['eventOwnerTeamId']
                event_type = row['typeDescKey']

                force_type = get_force(situation_code, event_owner_team_id, game['homeTeam']['id'], game['awayTeam']['id'])

                empty_net = goalie_id is None or (event_owner_team_id == game['homeTeam']['id'] and int(situation_code[0]) == 0) or (event_owner_team_id != game['homeTeam']['id'] and int(situation_code[3]) == 0)

                team = home_team if event_owner_team_id == game['homeTeam']['id'] else away_team

                plays_data.append({
                    'ID':game_id,
                    'Sort Order':sort_order,
                    'Time': time_in_period,
                    'Period': period_number,
                    'Event ID': event_id,
                    'Team': team,
                    'Type': event_type,
                    'Coordinates': (x_coord, y_coord),
                    'Shooter ID': shooter_id,
                    'Goalie ID': goalie_id,
                    'Shot Type': shot_type,
                    'Empty Net': empty_net,
                    'Force Type': force_type
                })

            # Créer la DataFrame finale
            df_final = pd.DataFrame(plays_data)

            # Créer le nom du fichier en utilisant le format spécifié
            filename = f'season_{year}_{game_id}.csv'
            file_path = os.path.join(output_dir, filename)

            # Sauvegarder la DataFrame dans un fichier CSV
            df_final.to_csv(file_path, index=False)
            print(f'Sauvegardé: {file_path}')


def load_dataframes(year_list):
    """
    Charge les DataFrames pour les années spécifiées et les concatène.

    :param year_list: Une liste d'années à charger.
    :return: Un dictionnaire où la clé est l'année et la valeur est la DataFrame concaténée pour cette année.
    """
    df_dict = {}

    for year in year_list:
        # Initialiser une liste pour stocker les DataFrames de l'année
        dfs = []

        files = [f for f in os.listdir("dataframe") if f.startswith(f'season_{year}_') and f.endswith(".csv")]

        files.sort()
        for filename in files:
            file_path = os.path.join('dataframe', filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

        # Concatenation des DataFrames pour l'année
        if dfs:
            df_year = pd.concat(dfs, ignore_index=True)
            df_dict[year] = df_year
        else:
            print(f"Aucun DataFrame trouvé pour l'année {year}.")

    return df_dict

