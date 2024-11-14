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
                'x_coord': details.get('xCoord'),
                'y_coord': details.get('yCoord'),
                'event_type': event_type,
                'empty_net': 1 if details.get('emptyNet') else 0  # Assume NaN as 0
            }
            shots_data.append(shot_info)
    
    return pd.DataFrame(shots_data)

def calculate_shot_angle(coord_str):
    """
    Calculates the angle of a shot relative to the goal in degrees.
    
    :param coord_str: String representing the shot coordinates "(x, y)"
    :return: The shot angle in degrees, or NaN if coordinates are invalid.
    """
    if pd.isna(coord_str):
        return np.nan  
    
    match = re.match(r'\((-?\d+),\s*(-?\d+)\)', coord_str)
    if match:
        x = int(match.group(1))  
        y = int(match.group(2)) 
    else:
        return np.nan
    
    goal_x = -89 if x < 0 else 89
    
    angle_radians = math.atan(y / (x - goal_x)) if x != goal_x else np.nan
    angle_degrees = math.degrees(angle_radians) if angle_radians is not None else np.nan
    
    return angle_degrees

def add_features(df):
    """
    Adds only the required features for shot analysis.
    
    :param df: DataFrame containing shot information.
    :return: DataFrame containing only the selected features.
    """
    # Calculate distance and angle, set goal flag, and keep only relevant columns
    df['shot_distance'] = df.apply(lambda row: simple_visualisation.calculate_shot_distance(f"({row['x_coord']}, {row['y_coord']})"), axis=1)
    df['shot_angle'] = df.apply(lambda row: calculate_shot_angle(f"({row['x_coord']}, {row['y_coord']})"), axis=1)
    df['is_goal'] = df['event_type'].apply(lambda x: 1 if x == 'goal' else 0)
    
    # Keep only the columns needed for analysis
    return df[['shot_distance', 'shot_angle', 'is_goal', 'empty_net']]

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
