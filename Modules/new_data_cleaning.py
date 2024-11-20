import pandas as pd
import math
import numpy as np

# Fonction pour convertir le temps en secondes
def time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except (ValueError, AttributeError):
        return None

# Calculer la distance de tir
def calculate_shot_distance(coord_str):
    if not coord_str or coord_str == 'None':  # Vérifie si la chaîne est vide ou invalide
        return None
    try:
        x, y = map(float, coord_str.strip('()').split(','))
        goal_x = 89 if x > 0 else -89
        return math.sqrt((x - goal_x) ** 2 + y ** 2)
    except Exception:
        return None

# Calculer l'angle de tir
def calculate_shot_angle(coord_str):
    if not coord_str or coord_str == 'None':  # Vérifie si la chaîne est vide ou invalide
        return None
    try:
        x, y = map(float, coord_str.strip('()').split(','))
        goal_x = 89 if x > 0 else -89
        angle_radians = math.atan2(y, (goal_x - x))
        return math.degrees(angle_radians)
    except Exception:
        return None

# Calculer la distance depuis le dernier événement
def calculate_distance(x1, y1, x2, y2):
    if any(pd.isna([x1, y1, x2, y2])):  # Si une des coordonnées est NaN, retourner NaN
        return np.nan
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fonction principale pour transformer le DataFrame
def transform_dataframe(df):
    # Conversion du temps en secondes
    df['Game seconds'] = df['Time'].apply(time_to_seconds)

    # Renommer la colonne `Period` en `Game period`
    df['Game period'] = df['Period']

    # Séparation des coordonnées en colonnes `x_coord` et `y_coord`
    df[['x_coord', 'y_coord']] = df['Coordinates'].str.extract(r'\((-?\d+),\s*(-?\d+)\)').astype(float)

    # Calcul des distances et angles de tir
    df['Shot distance'] = df['Coordinates'].apply(calculate_shot_distance)
    df['Shot angle'] = df['Coordinates'].apply(calculate_shot_angle)

    # Renommer `Shot Type` en `Shot type`
    df['Type'] = df['Shot Type']

    # Dernier type d'événement (Last event type)
    df['Last event type'] = df['Type'].shift(1)

    # Coordonnées du dernier événement
    df['Last x_coord'] = df['x_coord'].shift(1)
    df['Last y_coord'] = df['y_coord'].shift(1)

    # Temps écoulé depuis le dernier événement
    df['Time since last event'] = df['Game seconds'] - df['Game seconds'].shift(1)

    # Distance depuis le dernier événement
    df['Distance from last event'] = df.apply(
        lambda row: calculate_distance(row['x_coord'], row['y_coord'], row['Last x_coord'], row['Last y_coord']), axis=1
    )

    # Créer le DataFrame final avec les colonnes nécessaires
    df_final = df[[
        'ID', 'Game seconds', 'Game period', 'x_coord', 'y_coord',
        'Shot distance', 'Shot angle', 'Type',
        'Last event type', 'Last x_coord', 'Last y_coord',
        'Time since last event', 'Distance from last event'
    ]]
    
    return df_final

def add_advanced_features(df):
    """
    Ajoute des caractéristiques avancées au DataFrame, incluant les rebonds, le changement d'angle,
    et la vitesse. Prend également en compte le changement de période.
    
    :param df: DataFrame contenant les informations des tirs.
    :return: DataFrame enrichi avec les nouvelles caractéristiques.
    """
    # Ajouter les colonnes des événements précédents (décalage de 1 ligne)
    df['prev_event_type'] = df['Type'].shift(1)
    df['prev_x_coord'] = df['x_coord'].shift(1)
    df['prev_y_coord'] = df['y_coord'].shift(1)
    df['prev_shot_angle'] = df['Shot angle'].shift(1)
    df['prev_game_seconds'] = df['Game seconds'].shift(1)
    df['prev_period'] = df['Game period'].shift(1)  # Ajout de la période précédente

    # Rebond : Vérifie si le dernier événement était aussi un tir
    shot_types = ["shot-on-goal", "goal", "blocked-shot", "missed-shot"]
    df['Rebound'] = df['prev_event_type'].apply(lambda x: x in shot_types)

    # Changement d'angle de tir : Différence d'angle si c'est un rebond
    def calculate_angle_change(row):
        if row['Rebound']:
            return abs(row['Shot angle'] - row['prev_shot_angle'])
        return 0

    df['Angle change'] = df.apply(calculate_angle_change, axis=1)

    # Temps écoulé depuis le dernier événement (en secondes), en prenant en compte le changement de période
    def calculate_time_since_last_event(row):
        if row['Game period'] != row['prev_period']:  # Si la période a changé
            return 0  # Recommencer à zéro si la période change
        return row['Game seconds'] - row['prev_game_seconds']
    
    df['time_since_last_event'] = df.apply(calculate_time_since_last_event, axis=1)

    # Distance depuis l'événement précédent
    def calculate_distance_from_last(row):
        if pd.isna(row['prev_x_coord']) or pd.isna(row['prev_y_coord']):
            return 0
        return math.sqrt((row['x_coord'] - row['prev_x_coord']) ** 2 + 
                         (row['y_coord'] - row['prev_y_coord']) ** 2)

    df['distance_from_last_event'] = df.apply(calculate_distance_from_last, axis=1)

    # Vitesse : Distance divisée par le temps écoulé
    def calculate_speed(row):
        if row['time_since_last_event'] > 0:
            return row['distance_from_last_event'] / row['time_since_last_event']
        return 0

    df['Speed'] = df.apply(calculate_speed, axis=1)

    # Retourner le DataFrame avec les nouvelles colonnes ajoutées
    return df
