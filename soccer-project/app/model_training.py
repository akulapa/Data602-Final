# -*- coding: utf-8 -*-
"""
@author: Pavan Akula, Nnaemezue Obi-eyisi, Ilya Kats
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from app import app, db
from .models import Team, Player, Country, League, Team_Attributes, Player_Attributes, Match
from config import MATCHESCSV, MODELPICKLE, FEATURESPICKLE
import pickle

def getHomeFeatures(teamID, season, baseDf):
    '''Returns one entry corresponding to representative home features of a team in a season.'''
    allMatches = baseDf[(baseDf.home_team_api_id==teamID) & (baseDf.season==season)]
    lastMatch = allMatches.sort_values(by = 'date', ascending = False)[:1]
    matchCount = len(allMatches)
    if matchCount==0:
        return []
    lastMatch['home_prev'] = lastMatch['result']
    wins = len(allMatches[allMatches.result==1.0])
    lastMatch['home_win_rate'] = (wins/matchCount)*100
    lastMatch['home_play_passing'] = sum(allMatches.home_play_passing)/matchCount
    lastMatch['home_aggression'] = sum(allMatches.home_aggression)/matchCount
    lastMatch['home_team_width'] = sum(allMatches.home_team_width)/matchCount
    lastMatch['category'] = 'home'
    lastMatch = lastMatch[['home_team_api_id', 'season', 'category', 
                           'home_ranking', 'home_goalie_ranking', 
                           'home_players', 'home_goalie', 
                           'home_prev', 'home_win_rate', 
                           'home_play_passing', 'home_aggression', 'home_team_width']]
    lastMatch.columns = ['team_id', 'season', 'category', 
                         'ranking', 'goalie_ranking', 'players', 'goalie', 
                         'prev', 'win_rate', 'play_passing', 'aggression', 'team_width']
    return lastMatch

def getAwayFeatures(teamID, season, baseDf):
    '''Returns one entry corresponding to representative away features of a team in a season.'''
    allMatches = baseDf[(baseDf.away_team_api_id==teamID) & (baseDf.season==season)]
    lastMatch = allMatches.sort_values(by = 'date', ascending = False)[:1]
    matchCount = len(allMatches)
    if matchCount==0:
        return []
    lastMatch['away_prev'] = lastMatch['result']
    wins = len(allMatches[allMatches.result==1.0])
    lastMatch['away_win_rate'] = (wins/matchCount)*100
    lastMatch['away_play_passing'] = sum(allMatches.away_play_passing)/matchCount
    lastMatch['away_aggression'] = sum(allMatches.away_aggression)/matchCount
    lastMatch['away_team_width'] = sum(allMatches.away_team_width)/matchCount
    lastMatch['category'] = 'away'
    lastMatch = lastMatch[['away_team_api_id', 'season', 'category', 
                           'away_ranking', 'away_goalie_ranking', 
                           'away_players', 'away_goalie', 
                           'away_prev', 'away_win_rate', 
                           'away_play_passing', 'away_aggression', 'away_team_width']]
    lastMatch.columns = ['team_id', 'season', 'category', 
                         'ranking', 'goalie_ranking', 'players', 'goalie', 
                         'prev', 'win_rate', 'play_passing', 'aggression', 'team_width']
    return lastMatch
    

def getHomeWinPCT(teamID, season, matchDf):
    '''Returns home match winning percentage for previous season.'''
    if season=='2008/2009':
        return np.nan
    prevSeason = {'2009/2010':'2008/2009',
                  '2010/2011':'2009/2010',
                  '2011/2012':'2010/2011',
                  '2012/2013':'2011/2012',
                  '2013/2014':'2012/2013',
                  '2014/2015':'2013/2014',
                  '2015/2016':'2014/2015'}
    matches = len(matchDf[(matchDf.home_team_api_id==teamID) & (matchDf.season==prevSeason[season])])
    if matches==0:
        return np.nan
    wins = len(matchDf[(matchDf.home_team_api_id==teamID) & (matchDf.season==prevSeason[season]) & (matchDf.result==1)])
    return (wins/matches)*100

def getAwayWinPCT(teamID, season, matchDf):
    '''Returns away match winning percentage for previous season.'''
    if season=='2008/2009':
        return np.nan
    prevSeason = {'2009/2010':'2008/2009',
                  '2010/2011':'2009/2010',
                  '2011/2012':'2010/2011',
                  '2012/2013':'2011/2012',
                  '2013/2014':'2012/2013',
                  '2014/2015':'2013/2014',
                  '2015/2016':'2014/2015'}
    matches = len(matchDf[(matchDf.away_team_api_id==teamID) & (matchDf.season==prevSeason[season])])
    if matches==0:
        return np.nan
    wins = len(matchDf[(matchDf.away_team_api_id==teamID) & (matchDf.season==prevSeason[season]) & (matchDf.result==1)])
    return (wins/matches)*100

def getPrevResult(teamID, asOfDate, matchDf):
    ''' Returns result of previous game for a given team.'''
    homeMatches = matchDf[matchDf.home_team_api_id==teamID]
    awayMatches = matchDf[matchDf.away_team_api_id==teamID]
    allMatches = pd.concat([homeMatches, awayMatches])
    prevMatch = allMatches[allMatches.date < asOfDate].sort_values(by = 'date', ascending = False)[:1]
    if len(prevMatch)==0:
        return -99999
    else:
        return prevMatch['result'].values[0]
    
def getPlayerRanking(playerID, asOfDate, playerAttribDf):
    ''' Returns player ranking (one ID) or average of ranking (list of IDs).'''
    if type(playerID)==int:
        playerID=[playerID]
    ranking = []
    for pl in playerID:
        # Get the most recent available attributes
        playerAttribs = playerAttribDf[playerAttribDf.player_api_id==pl]
        currentAttribs = playerAttribs[playerAttribs.date <= asOfDate].sort_values(by = 'date', ascending = False)[:1]
        if len(currentAttribs)==0:
            ranking += [0]
        else:
            ranking += [currentAttribs.overall_rating.values[0]]
        return sum(ranking)/len(ranking)

def formatHomePlayers(row):
    return list(map(int, row[['home_player_2','home_player_3',
                              'home_player_4','home_player_5','home_player_6',
                              'home_player_7','home_player_8','home_player_9',
                              'home_player_10','home_player_11']].values.tolist()))
def formatAwayPlayers(row):
    return list(map(int, row[['away_player_2','away_player_3',
                              'away_player_4','away_player_5','away_player_6',
                              'away_player_7','away_player_8','away_player_9',
                              'away_player_10','away_player_11']].values.tolist()))

def getTeamName(teamID, teamDf):
    ''' Returns team's abbreviated name.'''
    return teamDf[teamDf.team_api_id==teamID].team_short_name.values[0]

def getPlayerHeight(playerID, playerDf):
    ''' Returns player height (if one ID) or average (if multiple IDs).'''
    if type(playerID)==int:
        playerID=[playerID]
    height = []
    for pl in playerID:
        height += [playerDf[playerDf.player_api_id==pl].height.values[0]]
    return sum(height)/len(height)

def getPlayerWeight(playerID, playerDf):
    ''' Returns player weight (if one ID) or average (if multiple IDs).'''
    if type(playerID)==int:
        playerID=[playerID]
    weight = []
    for pl in playerID:
        weight += [playerDf[playerDf.player_api_id==pl].weight.values[0]]
    return sum(weight)/len(weight)

def getTeamAttribute(teamID, asOfDate, attrib, teamAttribDf):
    ''' Returns team's attribute.'''
    teamAttribs = teamAttribDf[teamAttribDf.team_api_id==teamID]
    currentAttribs = teamAttribs[teamAttribs.date <= asOfDate].sort_values(by = 'date', ascending = False)[:1]
    if len(currentAttribs)==0:
        return np.nan
    else:
        return currentAttribs[attrib].values[0]

# --------------------------------------------------------------------
# READ DATA FROM DATABASE TO DATA FRAME
# --------------------------------------------------------------------

def generate_featurepickle():

    #sqliteFile = "C:\\Temp\\CUNY\\data602-final\\app\\database\\database.sqlite"
    #conn = sqlite3.connect(sqliteFile)
    #
    ## Match info
    #query = ("SELECT match_api_id, home_team_api_id, away_team_api_id, "+
    #         "       season, stage, home_team_goal, away_team_goal, date, "+
    #         "       home_player_1, home_player_2, home_player_3, "+
    #         "       home_player_4, home_player_5, home_player_6, "+
    #         "       home_player_7, home_player_8, home_player_9, "+
    #         "       home_player_10, home_player_11, "+
    #         "       away_player_1, away_player_2, away_player_3, "+
    #         "       away_player_4, away_player_5, away_player_6, "+
    #         "       away_player_7, away_player_8, away_player_9, "+
    #         "       away_player_10, away_player_11 "+
    #         "FROM Match;")
    #matchDf = pd.read_sql_query(query, conn)
    
    matchDf = pd.DataFrame()
    result = Match.query.with_entities(Match.match_api_id, Match.home_team_api_id, Match.away_team_api_id,
                                        Match.season, Match.stage, Match.home_team_goal, Match.away_team_goal, Match.date,
                                        Match.home_player_1, Match.home_player_2, Match.home_player_3, Match.home_player_4, Match.home_player_5, 
                                        Match.home_player_6, Match.home_player_7, Match.home_player_8, Match.home_player_9, Match.home_player_10, 
                                        Match.home_player_11,
                                        Match.away_player_1, Match.away_player_2, Match.away_player_3, Match.away_player_4, Match.away_player_5, 
                                        Match.away_player_6, Match.away_player_7, Match.away_player_8, Match.away_player_9, Match.away_player_10, 
                                        Match.away_player_11).filter(Match.league_id == 1729).filter(Match.season == '2015/2016') #.all()
    for row in result:
        data = {'match_api_id': row.match_api_id, 'home_team_api_id': row.home_team_api_id, 'away_team_api_id': row.away_team_api_id,
                'season': row.season, 'stage': row.stage, 'home_team_goal': row.home_team_goal, 'away_team_goal': row.away_team_goal, 'date': row.date,
                'home_player_1': row.home_player_1, 'home_player_2': row.home_player_2, 'home_player_3': row.home_player_3, 'home_player_4': row.home_player_4, 
                'home_player_5': row.home_player_5, 'home_player_6': row.home_player_6, 'home_player_7': row.home_player_7, 'home_player_8': row.home_player_8, 
                'home_player_9': row.home_player_9, 'home_player_10': row.home_player_10, 'home_player_11': row.home_player_11, 
                'away_player_1': row.away_player_1, 'away_player_2': row.away_player_2, 'away_player_3': row.away_player_3, 'away_player_4': row.away_player_4, 
                'away_player_5': row.away_player_5, 'away_player_6': row.away_player_6, 'away_player_7': row.away_player_7, 'away_player_8': row.away_player_8, 
                'away_player_9': row.away_player_9, 'away_player_10': row.away_player_10, 'away_player_11': row.away_player_11
                }
        matchDf = matchDf.append(data, ignore_index=True)
    
    print(matchDf)
    # Player info
    #query = ("SELECT player_api_id, birthday, height, weight "+
    #         "FROM Player;")
    #playerDf = pd.read_sql_query(query, conn)
    playerDf = pd.DataFrame()
    result = Player.query.with_entities(Player.player_api_id, Player.birthday, Player.height, Player.weight).all()
    for row in result:
        data = {'player_api_id': row.player_api_id, 'birthday': row.birthday, 'height': row.height, 'weight': row.weight}
        playerDf = playerDf.append(data, ignore_index=True)
    
    # Player attributes
    #query = ("SELECT player_api_id, date, overall_rating "+
    #         "FROM Player_Attributes;")
    #playerAttribDf = pd.read_sql_query(query, conn)
    
    playerAttribDf = pd.DataFrame()
    result = Player_Attributes.query.with_entities(Player_Attributes.player_api_id, Player_Attributes.date, Player_Attributes.overall_rating).all()
    for row in result:
        data = {'player_api_id': row.player_api_id, 'date': row.date, 'overall_rating': row.overall_rating}
        playerAttribDf = playerAttribDf.append(data, ignore_index=True)
    
    # Team info
    #query = ("SELECT team_api_id, team_short_name "+
    #         "FROM Team;")
    #teamDf = pd.read_sql_query(query, conn)
    
    teamDf = pd.DataFrame()
    result = Team.query.with_entities(Team.team_api_id, Team.team_short_name).all()
    for row in result:
        data = {'team_api_id': row.team_api_id, 'team_short_name': row.team_short_name}
        teamDf = teamDf.append(data, ignore_index=True)
    
    # Team attributes
    #query = ("SELECT team_api_id, date, "+
    #         "       buildUpPlaySpeed, buildUpPlayPassing, "+
    #         "       chanceCreationPassing, chanceCreationCrossing, "+
    #         "       chanceCreationShooting, "+
    #         "       defencePressure, defenceAggression, defenceTeamWidth "+
    #         "FROM Team_Attributes;")
    #teamAttribDf = pd.read_sql_query(query, conn)
    
    teamAttribDf = pd.DataFrame()
    result = Team_Attributes.query.with_entities(Team_Attributes.team_api_id, Team_Attributes.buildUpPlaySpeed, Team_Attributes.buildUpPlayPassing, 
                                                 Team_Attributes.chanceCreationPassing, Team_Attributes.chanceCreationCrossing, 
                                                 Team_Attributes.chanceCreationShooting, Team_Attributes.defencePressure, Team_Attributes.defenceAggression,
                                                 Team_Attributes.defenceTeamWidth).all()
    for row in result:
        data = {'team_api_id': row.team_api_id, 'buildUpPlaySpeed': row.buildUpPlaySpeed, 'buildUpPlayPassing': row.buildUpPlayPassing, 
                'chanceCreationPassing': row.chanceCreationPassing, 'chanceCreationCrossing': row.chanceCreationCrossing, 
                'chanceCreationShooting': row.chanceCreationShooting, 'defencePressure': row.defencePressure, 'defenceAggression': row.defenceAggression,
                'defenceTeamWidth': row.defenceTeamWidth}
        teamAttribDf = teamAttribDf.append(data, ignore_index=True)
    
    #conn.close()
    #del conn
    
    # --------------------------------------------------------------------
    # MODIFY DATA FOR TRAINING / GET FEATURES and LABEL
    # --------------------------------------------------------------------
    # Match outcome - 1-Win by home team, 2-Draw, 3-Loss by home team
    matchDf.loc[matchDf.home_team_goal >  matchDf.away_team_goal, 'result'] = 1
    matchDf.loc[matchDf.home_team_goal == matchDf.away_team_goal, 'result'] = 2
    matchDf.loc[matchDf.home_team_goal <  matchDf.away_team_goal, 'result'] = 3
    
    matchDf = matchDf.dropna()
    
    # Average ranking of players
    matchDf['home_ranking'] = matchDf.apply (lambda row: getPlayerRanking(formatHomePlayers(row), row['date'], playerAttribDf), axis=1)
    matchDf['away_ranking'] = matchDf.apply (lambda row: getPlayerRanking(formatAwayPlayers(row), row['date'], playerAttribDf), axis=1)
    # Ranking of goalies
    matchDf['home_goalie_ranking'] = matchDf.apply (lambda row: getPlayerRanking(int(row['home_player_1']), row['date'], playerAttribDf), axis=1)
    matchDf['away_goalie_ranking'] = matchDf.apply (lambda row: getPlayerRanking(int(row['away_player_1']), row['date'], playerAttribDf), axis=1)
    
    # Previous match result
    matchDf['home_prev'] = matchDf.apply (lambda row: getPrevResult(row['home_team_api_id'], row['date'], matchDf), axis=1)
    matchDf['away_prev'] = matchDf.apply (lambda row: getPrevResult(row['away_team_api_id'], row['date'], matchDf), axis=1)
    
    matchDf['home_win_rate'] = matchDf.apply (lambda row: getHomeWinPCT(row['home_team_api_id'], row['season'], matchDf), axis=1)
    matchDf['away_win_rate'] = matchDf.apply (lambda row: getAwayWinPCT(row['away_team_api_id'], row['season'], matchDf), axis=1)
    
    # Average height and weight of players
    #matchDf['home_height'] = matchDf.apply (lambda row: getPlayerHeight(formatHomePlayers(row), playerDf), axis=1)
    #matchDf['away_height'] = matchDf.apply (lambda row: getPlayerHeight(formatAwayPlayers(row), playerDf), axis=1)
    #matchDf['home_weight'] = matchDf.apply (lambda row: getPlayerWeight(formatHomePlayers(row), playerDf), axis=1)
    #matchDf['away_weight'] = matchDf.apply (lambda row: getPlayerWeight(formatAwayPlayers(row), playerDf), axis=1)
    
    # Considered using team names instead of IDs
    # Does not seem to be necessary    
    #matchDf['home_team'] = matchDf.apply (lambda row: getTeamName(row['home_team_api_id'], teamDf), axis=1)
    #matchDf['away_team'] = matchDf.apply (lambda row: getTeamName(row['away_team_api_id'], teamDf), axis=1)
    
    # Team attributes:
    # Build-up play speed and passing
    matchDf['home_play_speed'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'buildUpPlaySpeed', teamAttribDf), axis=1)
    matchDf['home_play_passing'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'buildUpPlayPassing', teamAttribDf), axis=1)
    matchDf['away_play_speed'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'buildUpPlaySpeed', teamAttribDf), axis=1)
    matchDf['away_play_passing'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'buildUpPlayPassing', teamAttribDf), axis=1)
    # Chance creation - passing, crossing, shooting
    matchDf['home_creation_passing'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'chanceCreationPassing', teamAttribDf), axis=1)
    matchDf['home_creation_crossing'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'chanceCreationCrossing', teamAttribDf), axis=1)
    matchDf['home_creation_shooting'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'chanceCreationShooting', teamAttribDf), axis=1)
    matchDf['away_creation_passing'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'chanceCreationPassing', teamAttribDf), axis=1)
    matchDf['away_creation_crossing'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'chanceCreationCrossing', teamAttribDf), axis=1)
    matchDf['away_creation_shooting'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'chanceCreationShooting', teamAttribDf), axis=1)
    # Defence - aggression and team width
    matchDf['home_aggression'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'defenceAggression', teamAttribDf), axis=1)
    matchDf['home_team_width'] = matchDf.apply (lambda row: getTeamAttribute(row['home_team_api_id'], row['date'], 'defenceTeamWidth', teamAttribDf), axis=1)
    matchDf['away_aggression'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'defenceAggression', teamAttribDf), axis=1)
    matchDf['away_team_width'] = matchDf.apply (lambda row: getTeamAttribute(row['away_team_api_id'], row['date'], 'defenceTeamWidth', teamAttribDf), axis=1)
    
    #matchDf.to_csv('matchDf.csv')
    
    matchDf.to_csv(MATCHESCSV)
    # --------------------------------------------------------------------
    # SELECT FIELD FOR TRAINING
    # --------------------------------------------------------------------
    modelDf = matchDf[[
    #                   'home_team_api_id', 'away_team_api_id', 
                       'stage', 
                       'home_ranking', 'away_ranking',
                       'home_goalie_ranking', 'away_goalie_ranking', 
                       'home_prev', 'away_prev', 
                       'home_win_rate', 'away_win_rate',
    #                   'home_height', 'away_height', 
    #                   'home_weight', 'away_weight', 
    #                   'home_play_speed', 'away_play_speed', 
                       'home_play_passing',  'away_play_passing',  
    #                   'home_creation_passing', 'home_creation_crossing', 'home_creation_shooting', 
    #                   'away_creation_passing', 'away_creation_crossing', 'away_creation_shooting', 
                       'home_aggression', 'away_aggression', 
                       'home_team_width', 'away_team_width', 
                       'result']]
    modelDf = modelDf.dropna()
    #print(len(modelDf))
    
    # --------------------------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------------------------
    X = np.array(modelDf.drop(['result'], 1))
    y = np.array(modelDf['result'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #clf = svm.SVC()
    #clf = KNeighborsClassifier(n_neighbors=5)
    #clf = GaussianNB()
    #clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1), n_estimators=1000, random_state=1)
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    
    print(accuracy)
    
    
    # --------------------------------------------------------------------
    # OPTIMIZE DATA FOR APP
    # --------------------------------------------------------------------
    matchDf['home_goalie'] = matchDf['home_player_1'].astype(int)
    matchDf['away_goalie'] = matchDf['away_player_1'].astype(int)
    matchDf['home_players'] = matchDf.apply (lambda row: formatHomePlayers(row), axis=1)
    matchDf['away_players'] = matchDf.apply (lambda row: formatAwayPlayers(row), axis=1)
    
    # Data frame for manipulation
    baseDf = matchDf[['season', 'home_team_api_id', 'away_team_api_id', 'date', 
                      'home_goalie', 'away_goalie',
                      'home_players', 'away_players', 
                      'home_ranking', 'away_ranking',
                      'home_goalie_ranking', 'away_goalie_ranking', 
                      'home_prev', 'away_prev', 
                      'home_win_rate', 'away_win_rate',
                      'home_play_passing',  'away_play_passing',  
                      'home_aggression', 'away_aggression', 
                      'home_team_width', 'away_team_width', 
                      'result']]
    baseDf = baseDf.dropna()
    
    #baseDf['baseFlag'] = ''
    
    # List of available teams
    homeTeams = baseDf['home_team_api_id'].unique()
    awayTeams = baseDf['away_team_api_id'].unique()
    allTeams = np.unique(np.concatenate([homeTeams, awayTeams]))
    
    # Features to save
    features = pd.DataFrame(columns=['team_id', 'season', 'category', 
                                     'ranking', 'goalie_ranking', 
                                     'players', 'goalie', 
                                     'prev', 'win_rate', 
                                     'play_passing', 'aggression', 'team_width'])
    
    # Loop through teams and season
    for team in allTeams: 
        for season in baseDf.season.unique():
            teamFeatures = getHomeFeatures(team, season, baseDf) 
            if len(teamFeatures) > 0:
                features = pd.concat([features, teamFeatures])
            teamFeatures = getAwayFeatures(team, season, baseDf) 
            if len(teamFeatures) > 0:
                features = pd.concat([features, teamFeatures])
    
    # Save for app use
    #features.to_pickle('features.pickle')
    
    features.to_pickle(FEATURESPICKLE)
    
    # Test
    #savedFeatures = pd.read_pickle('features.pickle')
    savedFeatures = pd.read_pickle(FEATURESPICKLE)
    
    print(len(savedFeatures))
    return 'Completed'
