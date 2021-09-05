from fantasy import positions, relevant_stats

import pandas as pd
import numpy as np

def last_n_weeks(df, stub, career_game, n):
    """
    Given the player's data stored in df, find the stats for the player's last n games 
    prior to career_game if they exist. Wrapping to previous year if necessary. If n > game
    will return whatever games are available so that returned df is smaller than expected.
    
    df: dataframe of player stats. Must have columns 'stub', 'career_game'
    stub: stub from pro football reference, guarenteed to be unique for each player
    career_game: career_game to look back from
    n: The number of weeks to look back. 
    
    Returns: last n weeks for the player before the given date
    """
    last_n = df[(df['stub'] == stub) & (df['career_game'] >= career_game - n) & (df['career_game'] < career_game)]
    
    # append 0 values to fill in missing values if career_game < n. Could maybe
    # add college stats but that would be a later addition
    if career_game < n:
        zero_rows = pd.DataFrame([[0] * last_n.shape[1]] * (n - career_game), columns=last_n.columns)
        return zero_rows.append(last_n, ignore_index=True)
    
    return last_n

def construct_model_vector(df, n):
    """
    Convert a dataframe to an array of numpy vectors which are of the form
    [(1-hot encoding of position), (game stats for n games leading up to this
      one for a given player)]. If there are p positions and s stats this
      vector will be of dimension p + s * n.
      
      df: dataframe of game values
      n: number of games to include in analysis
      
      Returns: 
    """
    data = []
    years = []
    
    for game in df.iterrows():
        pos = np.zeros(len(positions.keys()))
        pos[positions[game[1]['pos']]] = 1
        age = np.array([game[1]['age']])
        
        prev_games = last_n_weeks(df, game[1]['stub'], int(game[1]['career_game']), n)
        games_vec = prev_games[relevant_stats].to_numpy().flatten()
        final_vec = np.concatenate((pos, age, games_vec))
        
        data.append(final_vec)
        years.append(game[1]['year'])
    
    return np.stack(data), years
    