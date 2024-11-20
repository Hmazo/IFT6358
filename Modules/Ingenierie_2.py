from Modules.Ingenierie import calculate_shot_angle
from Modules.simple_visualisation import calculate_shot_distance
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_distance(x1, y1, x2, y2):
    """
    Calculates the distance between two points (x1, y1) and (x2, y2).
    
    :param x1, y1: Coordinates of the first point.
    :param x2, y2: Coordinates of the second point.
    :return: The Euclidean distance between the two points, or NaN if any coordinate is invalid.
    """
    # Check if any coordinate is missing
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return np.nan
    # Calculate and return the Euclidean distance
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_game_seconds(period, period_time):
    """
    Calculates the total seconds elapsed since the start of the game.
    
    :param period: Period number (1, 2, 3, or overtime periods).
    :param period_time: Time in the period as "MM:SS".
    :return: Total game seconds elapsed.
    """
    # Convert minutes and seconds from period_time
    minutes, seconds = map(int, period_time.split(':'))
    # Calculate total seconds including all previous periods
    return (period - 1) * 1200 + minutes * 60 + seconds

def extract_shots_data(game_data, season):
    """
    Extracts basic shot data without considering previous events.
    
    :param game_data: Dictionary containing the game's data.
    :param season: Season to which the game belongs.
    :return: DataFrame containing basic shot information.
    """
    shots_data = []

    for event in game_data['plays']:
        event_type = event['typeDescKey']
        if event_type in ["shot-on-goal", "goal", "blocked-shot", "missed-shot"]:
            details = event.get('details', {})
            x_coord, y_coord = details.get('xCoord'), details.get('yCoord')
            coord_str = f"({x_coord}, {y_coord})"  # Convert coordinates into string
            
            # Extract goalie ID, event owner team ID, and situation code from details
            goalie_id = details.get('goalieInNetId')
            event_owner_team_id = details.get('eventOwnerTeamId')
            situation_code = event.get('situationCode', "0000")  # Default to "0000" if missing
            home_team_id = game_data['homeTeam']['id']
            away_team_id = game_data['awayTeam']['id']
            
            # Calculate empty_net status
            empty_net = (
                goalie_id is None or 
                (event_owner_team_id == home_team_id and int(situation_code[0]) == 0) or 
                (event_owner_team_id != home_team_id and int(situation_code[3]) == 0)
            )

            # Gather relevant information for the shot
            shot_info = {
                'game_id': game_data['id'],
                'game_seconds': calculate_game_seconds(event['periodDescriptor']['number'], event['timeInPeriod']),
                'game_period': event['periodDescriptor']['number'],
                'x_coord': x_coord,
                'y_coord': y_coord,
                'shot_distance': calculate_shot_distance(coord_str),  # Calculate distance to goal
                'shot_angle': calculate_shot_angle(coord_str),       # Calculate angle relative to goal
                'shot_type': details.get('shotType'),               # Type of the shot (e.g., wrist, slap)
                'empty_net': 1 if empty_net else 0,                 # Assign 1 for empty net, 0 otherwise
            }
            shots_data.append(shot_info)
    
    # Convert the collected shot information to a DataFrame
    return pd.DataFrame(shots_data)


def extract_shots_with_previous_and_skater_info(game_data, season):
    """
    Extracts shot data with details about the previous event, skater counts, and attacking team information.
    
    :param game_data: Dictionary containing the game's data.
    :param season: Season to which the game belongs.
    :return: DataFrame containing enriched shot information.
    """
    shots_data = []
    last_event = {
        'type': None,  # Type of the last event
        'x': None,     # x-coordinate of the last event
        'y': None,     # y-coordinate of the last event
        'time': 0      # Game seconds at the last event
    }

    home_team = game_data['homeTeam']['name']['default']
    away_team = game_data['awayTeam']['name']['default']
    home_team_id = game_data['homeTeam']['id']
    away_team_id = game_data['awayTeam']['id']

    for event in game_data['plays']:
        event_type = event['typeDescKey']
        details = event.get('details', {})
        x_coord, y_coord = details.get('xCoord'), details.get('yCoord')
        coord_str = f"({x_coord}, {y_coord})"  # Convert coordinates into string
        period = event['periodDescriptor']['number']
        period_time = event['timeInPeriod']
        game_seconds = calculate_game_seconds(period, period_time)

        # Extract situation code and skater counts
        situation_code = event.get('situationCode', "0000")
        away_skaters = int(situation_code[1])  # Away skaters (non-goalies)
        home_skaters = int(situation_code[2])  # Home skaters (non-goalies)

        # Determine attacking team
        event_owner_team_id = details.get('eventOwnerTeamId')
        attacking_team_id = event_owner_team_id
        attacking_team_name = home_team if event_owner_team_id == home_team_id else away_team

        # Calculate empty_net status
        goalie_id = details.get('goalieInNetId')
        empty_net = (
            goalie_id is None or 
            (event_owner_team_id == home_team_id and int(situation_code[0]) == 0) or 
            (event_owner_team_id != home_team_id and int(situation_code[3]) == 0)
        )

        # Calculate time elapsed and distance from the last event
        time_elapsed = game_seconds - last_event['time']
        distance_from_last = calculate_distance(x_coord, y_coord,
                                                last_event['x'], last_event['y'])

        # Add shot information for valid shot events
        if event_type in ["shot-on-goal", "goal", "blocked-shot", "missed-shot"]:
            last_coord_str = f"({last_event['x']}, {last_event['y']})"  # Convert last event coordinates into string
            shot_info = {
                'game_id': game_data['id'],
                'game_seconds': game_seconds,
                'game_period': period,
                'x_coord': x_coord,
                'y_coord': y_coord,
                'shot_distance': calculate_shot_distance(coord_str),   # Calculate distance to goal
                'shot_angle': calculate_shot_angle(coord_str),         # Calculate angle relative to goal
                'shot_type': details.get('shotType'),
                'empty_net': 1 if empty_net else 0,                   # Assign 1 for empty net, 0 otherwise
                'last_event_type': last_event['type'],                # Type of the last event
                'last_event_x': last_event['x'],                      # x-coordinate of the last event
                'last_event_y': last_event['y'],                      # y-coordinate of the last event
                'time_since_last_event': time_elapsed,                # Time since the last event
                'distance_from_last_event': distance_from_last,       # Distance from the last event
                'friendly_skaters': home_skaters if event_owner_team_id == home_team_id else away_skaters,
                'opposing_skaters': away_skaters if event_owner_team_id == home_team_id else home_skaters,
                'attacking_team_id': attacking_team_id,
                'attacking_team_name': attacking_team_name,
                'home_team': home_team,
                'is_goal': 1 if event_type == "goal" else 0,          # Add is_goal column
            }
            shots_data.append(shot_info)

        # Update the last event details for the next iteration
        last_event.update({
            'type': event_type,
            'x': x_coord,
            'y': y_coord,
            'time': game_seconds
        })
    
    # Convert the collected shot information to a DataFrame
    return pd.DataFrame(shots_data)



def add_gameplay_features(df):
    """
    Adds gameplay features: rebound, shot angle change, and speed.
    
    :param df: DataFrame containing enriched shot data.
    :return: DataFrame with additional features.
    """
    # Feature 1: Check if the shot is a rebound
    df['rebound'] = df['last_event_type'].isin(['shot-on-goal'])

    # Feature 2: Calculate the shot angle change using the difference between current and previous shot angles
    df['shot_angle_change'] = df['shot_angle'] - df['shot_angle'].shift(1)
    df['shot_angle_change'] = df['shot_angle_change'].where(df['rebound'], 0)

    # Feature 3: Calculate the speed of the shot event (distance/time)
    def calculate_speed(row):
        if row['time_since_last_event'] > 0 and not pd.isna(row['distance_from_last_event']):
            return row['distance_from_last_event'] / row['time_since_last_event']
        return 0
    
    df['speed'] = df.apply(calculate_speed, axis=1)

    return df

def process_basic_shot_data(all_data):
    """
    Processes shot data for question 1 (basic features only).
    
    :param all_data: Dictionary where keys are seasons and values are lists of game data for each season.
    :return: Combined DataFrame of shots with basic features.
    """
    all_shots_data = []

    # Iterate through each season and its games
    for year, games in tqdm(all_data.items()):
        for game_data in games:
            # Extract basic shot information
            shots_df = extract_shots_data(game_data, f"{year}{year + 1}")
            all_shots_data.append(shots_df)

    # Combine all shot data into a single DataFrame
    combined_shots_df = pd.concat(all_shots_data, ignore_index=True)
    return combined_shots_df

def process_enriched_shot_data(all_data):
    """
    Processes shot data for questions 2 and onward (enriched features).
    
    :param all_data: Dictionary where keys are seasons and values are lists of game data for each season.
    :return: Combined DataFrame of shots with enriched features.
    """
    all_shots_data = []

    # Iterate through each season and its games
    for year, games in tqdm(all_data.items()):
        for game_data in games:
            # Extract enriched shot information
            shots_df = extract_shots_with_previous_and_skater_info(game_data, f"{year}{year + 1}")
            # Add additional gameplay features
            shots_df = add_gameplay_features(shots_df)
            all_shots_data.append(shots_df)

    # Combine all enriched shot data into a single DataFrame
    combined_shots_df = pd.concat(all_shots_data, ignore_index=True)
    return combined_shots_df
