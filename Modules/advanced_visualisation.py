import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import colors
from matplotlib.image import imread
from ipywidgets import IntSlider, Dropdown, interact





def determine_shot_coords(row):
    """
    Determines shot coordinates based on the period and whether the team is home.
    Filters out shots made from the team's own half.

    Args:
        row (Series): A row from the DataFrame containing shot data.
    
    Returns:
        tuple: Adjusted (x, y) coordinates if the shot is valid, otherwise None.
    """

    try:
        x, y = eval(row['Coordinates'])  # Parse the coordinates from the string
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
    except (SyntaxError, TypeError, NameError):
        return None

    is_home_team = row['Home']  
    period = row['Period']
    
    # Handle period logic
    if period in [1, 3]:  # Regular play, home team shoots on the left, away on the right
        if is_home_team:
            if x > 0:  # Invalid for home team in periods 1 or 3
                return None
        else:
            if x < 0:  # Invalid for away team in periods 1 or 3
                return None
    elif period in [2, 4]:  # Teams switch sides in the 2nd and 4th periods
        if is_home_team:
            if x < 0:  # Invalid for home team in periods 2 or 4
                return None
        else:
            if x > 0:  # Invalid for away team in periods 2 or 4
                return None
    elif period == 5:  # Shootout, ignore coordinates
        return None
    
    # Return valid coordinates, ensuring x is positive (as all valid shots are in the opponent's half)
    x = abs(x)
    return (y, x)


def adjust_shot_coordinates(df):
    """
    Adjusts shot coordinates based on the period and whether the team is home.
    Filters out shots from the team's own half of the rink.

    Args:
        df (DataFrame): DataFrame containing shot data.
    
    Returns:
        DataFrame: DataFrame with filtered and adjusted shot coordinates.
    """

    df['Adjusted Coordinates'] = df.apply(lambda row: determine_shot_coords(row), axis=1)
    df = df.dropna(subset=['Adjusted Coordinates'])
    return df




def aggregate_shot_locations(df,factor=1, grid_size=2):
    y_bins = np.arange(0, 101 , grid_size)  
    x_bins = np.arange(-42.5, 43.5 , grid_size)  

    df['xCoord'], df['yCoord'] = zip(*df['Adjusted Coordinates'])
    shot_counts, _, _ = np.histogram2d(df['xCoord'], df['yCoord'], bins=[x_bins, y_bins])
    total_games = df['ID'].nunique()
    return shot_counts/(total_games*factor)




def calculate_team_shot_rate(df, team_name):

    team_df = df[df['Team'] == team_name]
    team_shot_counts = aggregate_shot_locations(team_df)    
    return team_shot_counts


def calculate_shot_rate_difference(team_shot_rate, league_shot_rate, method='absolute'):

    if method == 'absolute':

        return team_shot_rate - league_shot_rate
    elif method == 'percentage':
        return (team_shot_rate - league_shot_rate) / league_shot_rate * 100
    else:
        raise ValueError("Method must be 'absolute' or 'percentage'.")



def plot_shot_rate(team, rate, year):
    # Prepare the figure
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Load the rink image and display it as the background
    img = imread('Data/vertical_rink.png')
    ax.imshow(img, extent=[-42.5, 42.5, 0, 100], aspect='auto', zorder=0)

    # Set extent for rink dimensions
    extent = [-42.5, 42.5, 0, 100]  # Set the extent to fit the rink dimension
    
    # Create grid for shot rates
    grid_size = 2
    x_bins = np.arange(-42.5, 43.5, grid_size)  # X-axis range: width of the rink
    y_bins = np.arange(0, 101, grid_size)       # Y-axis range: height of the offensive zone

    # Smooth the shot rates with a Gaussian filter
    shot_rate_smoothed = gaussian_filter(rate.T, sigma=1.5)
    max_abs_value = np.max(np.abs([np.min(shot_rate_smoothed), np.max(shot_rate_smoothed)]))
    shot_rate_smoothed = shot_rate_smoothed / max_abs_value

    # Custom colormap
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    )

    # Plot the contour
    contour = ax.contourf(
        x_bins[:-1], y_bins[:-1], shot_rate_smoothed, cmap=custom_cmap, alpha=0.7, zorder=1,
        levels=np.linspace(-1, 1, 10)
    )

    # Add colorbar
    fig.colorbar(contour, ax=ax, orientation="vertical", pad=0.05)

    # Set limits
    ax.set_xlim(-42.5, 42.5)
    ax.set_ylim(0, 100)

    # Add title with team and season info
    ax.set_title(f'{team} 5v5 Offence\n{year}-{year+1}, Regular Season\nShot Rates, Relative to League Average')

    # Customize appearance similar to the first image
    ax.set_xlabel('Distance from centre of rink (ft)')
    ax.set_ylabel('Distance from goal line (ft)')

    # Display the plot
    plt.show()
    


def widget(dataframes):

    
    year_list = dataframes.keys()
    year_dropdown = Dropdown(options=year_list, description='Select Year')
    rates = {}

    def update_year(selected_year):

        df = dataframes[selected_year]
        df['ID'] = df['ID'].astype(str)
        df = df[df['ID'].str[4:6] == "02"]

        df = adjust_shot_coordinates(df)
        league_average = aggregate_shot_locations(df, 2)
        teams = df['Team'].unique()
        
        # Calculate shot rates
        for team in teams:
            team_shot_rate = calculate_team_shot_rate(df, team)
            rates[team] = calculate_shot_rate_difference(team_shot_rate, league_average, method='absolute')

        # Update the team dropdown
        team_dropdown = Dropdown(options=rates.keys(), description='Select team')
        
        def update_team(selected_team):
            """
            Update the plot based on the selected team and year.
            """
            # Generate and plot the shot rate data for the selected team
            plot_shot_rate(selected_team, rates[selected_team], selected_year)
        # Interact with the team selection
        interact(update_team, selected_team=team_dropdown)

    # Interactive dropdown for year selection
    interact(update_year, selected_year=year_dropdown)



