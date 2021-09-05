import pytest
import pandas as pd

from fantasy import fantasy_points, stat_convs

mock_stat_convs = {
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

# Convert a list of fake stats to create a dictionary of player stats
def create_player(stats):
    return pd.DataFrame(stats)

# Corner cases
def test_empty_stats():
    player = create_player([{}])
    assert fantasy_points(player, mock_stat_convs)[0] == 0
    
def test_all_zero():
    player = create_player([{
        'pass_yds': 0,
        'rush_yds': 0,
        'rec_yds': 0,
        'rec': 0,
        'pass_td': 0,
        'rush_td': 0,
        'rec_td': 0,
        'two_pt_md': 0,
        'fumbles_lost': 0,
        'pass_int': 0,
        'kick_ret_td': 0,
        'punt_ret_td': 0
    }])
    assert fantasy_points(player, mock_stat_convs)[0] == 0

# Some representative test cases
def test_qb_score():
    player = create_player([{
        'pass_yds': 300,
        'pass_att': 20,
        'pass_td': 2,
        'rush_yds': 16,
        'rush_att': 2,
        'pass_int': 1,
    }])
    
    assert fantasy_points(player, mock_stat_convs)[0] == 19.6
    
def test_rb_score():
    player = create_player([{
        'rush_yds': 97,
        'rush_att': 22,
        'rec': 3,
        'rec_yds': 30,
        'rush_td': 1,
        'rec_td': 2,
    }])
    
    assert fantasy_points(player, mock_stat_convs)[0] == 32.2
    
def test_wr_score():
    player = create_player([{
        'rec': 6,
        'rec_yds': 98,
        'two_pt_md': 1,
        'fumbles_lost': 1,
    }])
    
    assert fantasy_points(player, mock_stat_convs)[0] == 12.8
        
