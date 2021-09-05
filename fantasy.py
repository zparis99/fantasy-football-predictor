import numpy as np

# Define constants for fantasy points
# stat fields and corresponding point per stat
stat_convs = {
    'pass_yds': 0.04,
    'rush_yds': 0.1,
    'rec_yds': 0.1,
    'rec': 0.5,
    'pass_td': 4,
    'rush_td': 6,
    'rec_td': 6,
    'two_pt_md': 2,
    'fumbles_lost': -2,
    'pass_int': -2,
    'kick_ret_td': 6,
    'punt_ret_td': 6
}

# Stats useful for modelling
relevant_stats = ['pass_yds', 'rush_yds', 'rec_yds', 'rec', 'pass_td', 'rush_td',
                   'rec_td', 'two_pt_md', 'fumbles_lost', 'pass_int', 'kick_ret_td', 
                   'punt_ret_td', 'pass_att', 'rush_att', 'fantasy_points']

# Used to map position code to integer
positions = {
    'QB': 0,
    'RB': 1,
    'WR': 2,
    'TE': 3,
}

# Calculate fantasy points from received stats
def fantasy_points(df, stat_convs=stat_convs):
    """
    Calculates a players fantasy points given their offensive stats for a week
    
    df: dataframe of player data. Assumed to have all of the columns i stat_convs
    stat_convs: dictionary of form 
        {
            stat_id: points per stat,
            ...
        }
        by default uses the variable in this file but can be overwritten
    """
    # Filled na values with 0
    accumulator = np.zeros(df.shape[0])
    for field in stat_convs.keys():
        accumulator += df.get(field, default=0) * stat_convs[field]
        
    return accumulator
