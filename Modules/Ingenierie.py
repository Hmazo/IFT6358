from Modules import simple_visualisation
import pandas as pd
import numpy as np
import re
import math

def extract_shots_data(game_data, season):
    """
    Extracts relevant shot information from game data.
    
    :param game_data: Dictionary containing the game's data.
    :param season: Season to which the game belongs.
    :return: DataFrame containing shot information with additional features.
    """
    shots_data = []
    
    for event in game_data['plays']:
        event_type = event['typeDescKey']
        if event_type in ["shot-on-goal", "goal", "blocked-shot", "missed-shot"]:
            details = event.get('details', {})
            shot_info = {
                'season': season,
                'event_id': event['eventId'],
                'time_in_period': event['timeInPeriod'],
                'period': event['periodDescriptor']['number'],
                'event_type': event_type,
                'x_coord': details.get('xCoord'),
                'y_coord': details.get('yCoord'),
                'shot_type': details.get('shotType'),
                'shooter_id': details.get('shootingPlayerId'),
                'goalie_id': details.get('goalieInNetId'),
                'team_id': details.get('eventOwnerTeamId'),
                'empty_net': 1 if details.get('emptyNet') else 0  # Assume NaN as 0
            }
            shots_data.append(shot_info)
    
    return pd.DataFrame(shots_data)

def calculate_shot_angle(coord_str):
    """
    Calcule l'angle d'un tir par rapport au but en degrés.
    
    :param coord_str: Chaîne représentant les coordonnées du tir sous la forme "(x, y)", par exemple "(85, -1)".
    :return: L'angle du tir en degrés, ou NaN si les coordonnées sont invalides.
    """
    
    # Vérifier si les coordonnées sont manquantes
    if pd.isna(coord_str):
        return np.nan  
    
    # Utiliser une expression régulière pour extraire les valeurs x et y de la chaîne de caractères
    match = re.match(r'\((-?\d+),\s*(-?\d+)\)', coord_str)
    if match:
        # Convertir les coordonnées en entiers
        x = int(match.group(1))  
        y = int(match.group(2)) 
    else:
        return np.nan  # Retourne NaN si le format de la chaîne est invalide
    
    # Déterminer le côté vers lequel le tir a été effectué (but gauche ou droit)
    goal_x = -89 if x < 0 else 89  # Position x du but en fonction du côté du terrain
    
    # Calculer l'angle en radians entre le tireur et le but
    # Utilisation de arctan pour obtenir l'angle entre la ligne du tir et la perpendiculaire au but
    # Calculer l'angle en radians
    angle_radians = math.atan(y / (x - goal_x)) if x != goal_x else np.nan
    
    # Convertir l'angle de radians en degrés pour faciliter l'interprétation
    angle_degrees = math.degrees(angle_radians) if angle_radians is not None else np.nan
    
    # Retourner l'angle en degrés
    return angle_degrees

def add_features(df):
    """
    Adds 'shot_distance', 'shot_angle', 'is_goal', and 'empty_net' features to a shot DataFrame.
    
    :param df: DataFrame containing shot information.
    :return: DataFrame enriched with the new features.
    """
    df['shot_distance'] = df.apply(lambda row: simple_visualisation.calculate_shot_distance(f"({row['x_coord']}, {row['y_coord']})"), axis=1)
    df['shot_angle'] = df.apply(lambda row: calculate_shot_angle(f"({row['x_coord']}, {row['y_coord']})"), axis=1)
    df['is_goal'] = df['event_type'].apply(lambda x: 1 if x == 'goal' else 0)
    return df

def process_loaded_games(all_data):
    """
    Processes loaded game data by extracting shots and adding features for each game.
    
    :param all_data: Dictionary where keys are seasons and values are lists of game data for each season.
    :return: Combined DataFrame of shots for all games and seasons.
    """
    all_shots_data = []

    # Iterate over each season and each game in that season
    for year, games in all_data.items():
        season = f"{year}{year + 1}"
        for game_data in games:
            shots_df = extract_shots_data(game_data, season)
            shots_df = add_features(shots_df)
            all_shots_data.append(shots_df)

    # Combine all shot data into a single DataFrame
    combined_shots_df = pd.concat(all_shots_data, ignore_index=True)
    return combined_shots_df


