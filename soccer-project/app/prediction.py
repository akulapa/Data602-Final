# -*- coding: utf-8 -*-
"""
@author: Pavan Akula, Nnaemezue Obi-eyisi, Ilya Kats
"""

import pickle
import pandas as pd
import numpy as np
from config import MODELPICKLE, FEATURESPICKLE

def loadModel():
    pickleFile = MODELPICKLE
    with open(pickleFile, 'rb') as handle:
        clf = pickle.load(handle)
    return clf

def loadBaseFeatures():
    pickleFile = FEATURESPICKLE
    features = pd.read_pickle(pickleFile)
    return features

def getPrediction(model, features, homeTeamID, awayTeamID, season='2015/2016'):
    '''
    Get a predicted outcome for given model and features of home and away teams.
    Returns: 
        - Outcome - 1-Home Win, 2-Draw, 3-Home Loss - single value
        - Associated probabilities - numpy array
        - Associated features - pandas data frame
    Return 0 for outcome if features are not available.
    '''
    homeFeatures = features[(features.team_id==homeTeamID) & (features.season==season) & (features.category=='home')]
    awayFeatures = features[(features.team_id==awayTeamID) & (features.season==season) & (features.category=='away')]
    allFeatures = pd.DataFrame(columns=['home_api_team_id', 'away_api_team_id', 'season', 
                                        'home_ranking', 'away_ranking', 
                                        'home_goalie_ranking', 'away_goalie_ranking', 
                                        'home_players', 'away_players', 
                                        'home_goalie', 'away_goalie', 
                                        'home_prev', 'away_prev', 
                                        'home_win_rate', 'away_win_rate', 
                                        'home_play_passing', 'away_play_passing', 
                                        'home_aggression', 'away_aggression', 
                                        'home_team_width', 'away_team_width'])
    if len(homeFeatures)!=1 or len(awayFeatures)!=1:
        return 0.0, np.array([33.3333, 33.3333, 33.3333]), allFeatures
    homeFeatures.columns = ['home_api_team_id', 'season', 'category', 
                            'home_ranking', 'home_goalie_ranking', 
                            'home_players', 'home_goalie', 
                            'home_prev', 'home_win_rate', 
                            'home_play_passing', 'home_aggression', 'home_team_width']
    homeFeatures = homeFeatures.drop(['category'], 1)
    homeFeatures.index = [0]
    awayFeatures.columns = ['away_api_team_id', 'season', 'category', 
                            'away_ranking', 'away_goalie_ranking', 
                            'away_players', 'away_goalie', 
                            'away_prev', 'away_win_rate', 
                            'away_play_passing', 'away_aggression', 'away_team_width']
    awayFeatures = awayFeatures.drop(['category', 'season'], 1)
    awayFeatures.index = [0]
    allFeatures = pd.concat([homeFeatures, awayFeatures], axis=1)
    allFeatures = allFeatures[['home_api_team_id', 'away_api_team_id', 'season', 
                               'home_ranking', 'away_ranking', 
                               'home_goalie_ranking', 'away_goalie_ranking', 
                               'home_players', 'away_players', 
                               'home_goalie', 'away_goalie', 
                               'home_prev', 'away_prev', 
                               'home_win_rate', 'away_win_rate', 
                               'home_play_passing', 'away_play_passing', 
                               'home_aggression', 'away_aggression', 
                               'home_team_width', 'away_team_width']]        
    modelFeatures = allFeatures[['home_ranking', 'away_ranking',
                                 'home_goalie_ranking', 'away_goalie_ranking', 
                                 'home_prev', 'away_prev', 
                                 'home_win_rate', 'away_win_rate',
                                 'home_play_passing',  'away_play_passing',  
                                 'home_aggression', 'away_aggression', 
                                 'home_team_width', 'away_team_width']]
    outcome = model.predict(modelFeatures)
    probs = model.predict_proba(modelFeatures)
    return outcome[0], probs[0], allFeatures
