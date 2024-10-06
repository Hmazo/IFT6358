import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

def shotsComparaison(dfs, year):
    """
    Fonction pour comparer le nombre total de tirs et le nombre total de buts par type de tir.
    dfs est une liste de DataFrames, chaque DataFrame correspondant à une saison.
    Lorsqu'on fait dfs[year], cela nous donne les données pour une saison spécifique.
    
    :param dfs: Liste des DataFrames pour chaque saison.
    :param year: L'année pour laquelle on veut effectuer la comparaison.
    """

    # Récupérer les données pour l'année spécifiée
    df = dfs[year]

    # Grouper les données par type de tir et calculer le nombre total de tirs et de buts
    shot_data = df.groupby('Shot Type').agg({'Type': 'count', 'Empty Net': 'sum'}).rename(columns={'Type': 'Total Shots', 'Empty Net': 'Total Goals'})

    # Créer un graphique pour superposer le nombre total de tirs et de buts
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Tracer le nombre total de tirs sur l'axe y principal
    ax2 = ax1.twinx()  # Deuxième axe y pour le nombre de buts
    shot_data['Total Shots'].sort_values(ascending=False).plot(kind='bar', ax=ax1, color='lightcoral', position=1, width=0.4, label='Total Shots')

    # Tracer le nombre total de buts sur l'axe y secondaire
    shot_data['Total Goals'].sort_values(ascending=False).plot(kind='bar', ax=ax2, color='skyblue', position=0, width=0.4, label='Total Goals')

    # Ajouter des titres et des étiquettes pour les axes
    ax1.set_title(f'Comparaison Types de Tirs - Season {year}')
    ax1.set_xlabel('Type de Tir')
    ax1.set_ylabel('Total des Tirs', color='lightcoral')
    ax2.set_ylabel('Total des Buts', color='skyblue')

    # Ajouter des légendes pour les deux axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')



def calculate_shot_distance(coord_str):
    """
    Calcule la distance d'un tir par rapport au but en fonction des coordonnées x et y.
    On suppose que le but est positionné à (-89, 0) ou (89, 0).
    
    :param coord_str: Chaîne représentant les coordonnées, par exemple "(85, -1)"
    :return: La distance du tir par rapport au but, ou NaN si les coordonnées sont invalides.
    """

    # Vérifier si les coordonnées sont manquantes
    if pd.isna(coord_str):
        return np.nan
    
    # Utiliser une expression régulière pour extraire les valeurs x et y de la chaîne de caractères
    match = re.match(r'\((-?\d+),\s*(-?\d+)\)', coord_str)
    if match:
        x = int(match.group(1))  # Extraire la coordonnée x
        y = int(match.group(2))  # Extraire la coordonnée y
    else:
        return np.nan  # Retourner NaN si le format est invalide
    
    # Vérifier si les coordonnées sont valides
    if x is not None and y is not None:
        # Déterminer vers quel but le tir a été effectué
        if x < 0:
            # Tir effectué vers le but à (-89, 0)
            goal_x = -89
        else:
            # Tir effectué vers le but à (89, 0)
            goal_x = 89
        
        # Calculer la distance entre la position du tir et le but
        return np.sqrt((x - goal_x)**2 + y**2)
    else:
        return np.nan  # Retourner NaN si les coordonnées sont manquantes
    


def distancegoal_relationship(dfs, year):
    """
    Visualise la relation entre la distance du tir et la probabilité qu'il devienne un but.
    dfs est une liste de DataFrames pour chaque saison.

    :param dfs: Liste des DataFrames pour chaque saison.
    :param year: L'année pour laquelle on veut analyser la relation distance-but.
    """
    
    # Récupérer les données pour l'année spécifiée
    df = dfs[year]
    
    # Appliquer la fonction pour calculer la distance des tirs à partir des coordonnées
    df['Shot Distance'] = df['Coordinates'].apply(calculate_shot_distance)

    # Supprimer les lignes avec des informations de distance manquantes si nécessaire
    df = df.dropna(subset=['Shot Distance'])

    # Calculer la probabilité de but en fonction de la distance du tir
    df['Goal'] = df['Type'] == 'goal'
    goal_probability = df.groupby('Shot Distance').agg({'Goal': 'mean', 'Shot Distance': 'count'}).rename(columns={'Goal': 'Probabilité de But', 'Shot Distance': 'Nombre de Tirs'})
    
    # Créer un graphique en points montrant la relation entre la distance du tir et la probabilité de but
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Shot Distance', y='Probabilité de But', data=goal_probability, size='Nombre de Tirs', legend=False)

    # Ajouter un titre et des étiquettes pour les axes
    plt.title(f'Probabilité de But par Distance du Tir (Saison {year})')
    plt.xlabel('Distance du Tir')
    plt.ylabel('Probabilité de But')

    # Afficher le graphique
    plt.show()


def distance_goalpercentage (dfs, year):

    # Récupérer les données pour l'année spécifiée
    df = dfs[year]
    
    # Appliquer la fonction pour calculer la distance des tirs à partir des coordonnées
    df['Shot Distance'] = df['Coordinates'].apply(calculate_shot_distance)

    # Supprimer les lignes avec des informations de distance manquantes si nécessaire
    df = df.dropna(subset=['Shot Distance'])

    # Créer des bins pour les distances (par exemple, des intervalles de 5 unités)
    bins = np.arange(0, 100, 5)  # Créer des intervalles de 5 unités entre 0 et 100
    df['Binned Distance'] = pd.cut(df['Shot Distance'], bins=bins)

    # Calculer la probabilité de but par type de tir et par distance binned
    df['Goal'] = df['Type'] == 'goal'
    shot_distance_data = df.groupby(['Shot Type', 'Binned Distance']).agg({'Goal': 'mean'}).rename(columns={'Goal': 'Goal Percentage'}).reset_index()

    # Créer une table pivot où chaque ligne est un type de tir et chaque colonne une distance binned
    shot_distance_pivot = shot_distance_data.pivot_table(index='Shot Type', columns='Binned Distance', values='Goal Percentage')

    # Remplacer les valeurs manquantes dans la table pivot (par exemple, avec 0 ou NaN)
    shot_distance_pivot = shot_distance_pivot.fillna(0)

    # Tracer la heatmap 
    plt.figure(figsize=(10, 6))
    sns.heatmap(shot_distance_pivot, cmap='Reds', annot=False, linewidths=0.5)

    # Ajouter un titre et des étiquettes pour les axes
    plt.title('Pourcentage de Buts par Distance (Binned) et Type de Tir (Heatmap)')
    plt.xlabel('Distance Binned du Tir')
    plt.ylabel('Type de Tir')

    # Afficher la heatmap
    plt.show()