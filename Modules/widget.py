import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ipywidgets import IntSlider, Dropdown, interact
from Modules import data




def plot_rink(game, event):
    """
    Plots the hockey rink and displays the location of a specific event on the rink.

    Args:
        game (dict): A dictionary containing game information, including game ID and season.
        event (dict): A dictionary containing event details, such as event coordinates 
                      ('xCoord', 'yCoord') and event description ('typeDescKey').
    
    The function uses the provided event data to plot the location of the event on the rink. 
    If the event has coordinates, it marks the position and labels it with a description.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    rink_img_path = 'Data/rink.png'
    rink_img = mpimg.imread(rink_img_path)
    
    # Display the hockey rink image within the provided coordinates
    ax.imshow(rink_img, extent=[-100, 100, -42.5, 42.5])
    
    # Check if event contains x and y coordinates to plot
    if 'details' in event:
        if 'xCoord' in event['details'] and 'yCoord' in event['details']:
            x, y = event['details']['xCoord'], event['details']['yCoord']
            # Plot event position on the rink
            plt.scatter(x, y, color='blue', s=100)
            # Add event description text at the plotted position
            plt.text(x + 5, y, event['typeDescKey'], fontsize=12, color='white', 
                     bbox=dict(facecolor='black', alpha=0.5))
    
    # Set rink boundaries and aspect ratio
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_title(f"Game {game['id']} - {game['season']}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def browse_events(data,game_idx, event_idx):
    """
    Displays information about a specific game and event, and visualizes the event on the rink.

    Args:
        game_idx (int): Index of the game in the dataset.
        event_idx (int): Index of the event within the game's play-by-play data.
    
    The function prints detailed information about the game (teams, score, shots on goal, overtime, 
    and shootout details) and then calls `plot_rink` to visualize the event on a rink plot.
    """
    # Fetch the specific game and event from the data
    game = data[game_idx]
    event = game['plays'][event_idx]
    
    # Print basic game information
    print("\tGame ID:", game['id'])
    print("\tSeason:", game['season'])
    print(f"\tDate and Time: {game['gameDate']} at {game['startTimeUTC']}")

    home_team = game['homeTeam']
    away_team = game['awayTeam']
    
    # Print teams, score, and shots on goal (SoG) statistics
    print("\n")
    print(f"\tTeams: {home_team['abbrev']} (Home) vs {away_team['abbrev']} (Away)")
    print(f"\tGoals: {home_team['score']} (Home) - {away_team['score']} (Away)")
    print(f"\tSoG: {home_team['sog']} (Home) - {away_team['sog']} (Away)")
    print("\n")
    # Check if the game went to overtime or shootout
    if game['otInUse']:
        print("\t\tOvertime: Yes")
    else:
        print("\t\tOvertime: No")
        
    if game['shootoutInUse']:
        print("\t\tShootout: Yes")
        print(f"\t\tSO Goals: {home_team.get('shootoutGoals', 'None')} (Home) - {away_team.get('shootoutGoals', 'None')} (Away)")
        print(f"\t\tSO Attempts: {home_team.get('shootoutAttempts', 'None')} (Home) - {away_team.get('shootoutAttempts', 'None')} (Away)")
    else:
        print("\t\tShootout: No")

    # Plot the event on the rink and print the event details
    plot_rink(game, event)
    print(event)


def filter(data,game_type):
    result = []
    for game in data:
        if game_type == str(game["id"])[4:6]:
            result.append(game)
    return result

def widget(year_list):
    """
    Function to load game data for the given years, set up a dropdown to select the year,
    and interactive sliders to browse games and events within the selected year.

    Args:
        year_list (list): A list of years from which the NHL data will be loaded.
        
    Returns:
        dict: A dictionary where each year maps to its loaded game data.
    """
    # Load the data using the load_data function
    all_data = data.load_data(year_list)  # Load each year and store in dict
    type_dict = {"Regular Season":"02","Play Offs":"03"}
    year_dropdown = Dropdown(options=year_list, description='Select Year')
    type_dropdown = Dropdown(options=type_dict.keys(), description='Select Type')
    game_slider = IntSlider(min=0, max=10, step=1, description='Game ID')
    event_slider = IntSlider(min=0, max=10, step=1, description='Event')


    def update_year(selected_year,selected_type):
        """
        Updates the game and event sliders based on the selected year.
        """
        year_games = all_data[selected_year]  # Get the data for the selected year
        games = filter(year_games,type_dict[selected_type])
        # Update the game slider based on the number of games in the selected year
        game_slider.max = len(games) - 1
        game_slider.value = 0  

        # Update the event slider for the first game initially
        event_slider.max = len(games[0]['plays']) - 1
        event_slider.value = 0  

        def update_plot(game_idx, event_idx):
            """
            Updates the event slider based on the selected game and displays the event.
            """
            event_slider.max = len(games[game_idx]['plays']) - 1  
            browse_events(games,game_idx, event_idx) 

        # Use the updated sliders interactively
        interact(update_plot, game_idx=game_slider, event_idx=event_slider)

    # Interactive dropdown for year selection
    interact(update_year, selected_year=year_dropdown, selected_type = type_dropdown)
