#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 2017

@authors: Ilya Kats, Nnaemezue Obi-Eyisi Pavan Akula
"""

from app import app, db
from .models import Team, Player, Country, League, Team_Attributes, Player_Attributes, Match
import pandas as pd
import datetime
import time
from sqlalchemy import func, desc
from sqlalchemy.sql.expression import label
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
from collections import OrderedDict
from .prediction import loadModel, loadBaseFeatures, getPrediction


def get_team_name(team):
    result = Team.query.with_entities(Team.team_long_name).filter(Team.team_api_id == team).first()
    teamName = ' '
    for row in result:
        teamName = row
    return(teamName)

def get_team_shortname(team):
    result = Team.query.with_entities(Team.team_short_name).filter(Team.team_api_id == team).first()
    teamName = ' '
    for row in result:
        teamName = row
    return(teamName)

def get_playername(player):
    player = int(player)
    result = Player.query.with_entities(Player.player_name).filter(Player.player_api_id == player).first()
    playerName = ' '
    for row in result:
        playerName = row
    return(playerName)

def get_teams():
    
    result = db.session.query(Team).all()
    df = pd.DataFrame()
    for row in result:
        data = {'Team ID': row.team_short_name, 'Team': row.team_long_name}
        df = df.append(data, ignore_index=True)
    return df

def get_prediction_detail(outcome, outcome_prob, matchFeatures, homeTeamID, awayTeamID):
    outcomeDf = pd.DataFrame()
    keyPlayersDf = pd.DataFrame()
    
    if (outcome==0):
        predOutcome = ['<b>Not Enough Data</b>','<b>Not Enough Data</b>']
    elif (outcome==1):
        predOutcome = ['<b>WIN</b>','<b>LOSS</b>']
    elif (outcome==2):
        predOutcome = ['<b>DRAW</b>','<b>DRAW</b>']
    else:
        predOutcome = ['<b>LOSS</b>','<b>WIN</b>']
        
    if (outcome>0):
        outcomeData = OrderedDict([ 
                                    ('Prediction', ['Prediction', 'Team', 'Ranking', 'Rate']),
                                    ('Home', [predOutcome[0], get_team_name(homeTeamID),
                                              "%.2f" % matchFeatures.home_ranking.values[0], 
                                              ("%.2f" % matchFeatures.home_win_rate.values[0]) + " %",
                                             ]),
                                    ('Away', [predOutcome[1], get_team_name(awayTeamID),
                                              "%.2f" % matchFeatures.away_ranking.values[0], 
                                              ("%.2f" % matchFeatures.away_win_rate.values[0]) + " %"
                                             ]),
                                    ('rid', list(range(0, 4)))
                                ])
        
        homeTeamList = matchFeatures.home_players.values[0]
        
        if matchFeatures.home_goalie.values[0] in homeTeamList:
            homeTeamList.remove(matchFeatures.home_goalie.values[0])
            
        homeTeamList.insert(0, matchFeatures.home_goalie.values[0])
        
        awayTeamList = matchFeatures.away_players.values[0]
        
        if matchFeatures.away_goalie.values[0] in awayTeamList:
            awayTeamList.remove(matchFeatures.away_goalie.values[0])

        awayTeamList.insert(0, matchFeatures.away_goalie.values[0])
        
        while len(homeTeamList) > len(awayTeamList):
            homeTeamList = homeTeamList[:-1]

        while len(awayTeamList) > len(homeTeamList):
            awayTeamList = awayTeamList[:-1]
        
        homeTeamNamesList = [get_playername(playerId) + ' (Goalie)' if idx == 0 else get_playername(playerId) for idx,playerId in enumerate(homeTeamList)]
        awayTeamNamesList = [get_playername(playerId) + ' (Goalie)' if idx == 0 else get_playername(playerId) for idx,playerId in enumerate(awayTeamList)]
        
        keyPlayersData = OrderedDict([
                                        ('Home', homeTeamNamesList),
                                        ('Away', awayTeamNamesList),
                                        ('HomePlayerId', homeTeamList),
                                        ('AwayPlayerId', awayTeamList),
                                        ('rid', list(range(0, len(homeTeamList))))

                             ])

        keyPlayersDf = pd.DataFrame.from_dict(keyPlayersData)
        
    else:
        outcomeData = OrderedDict([ 
                        ('Prediction', ['Prediction', 'Team', 'Ranking', 'Win Rate']),
                        ('Home', [predOutcome[0], get_team_name(homeTeamID),
                                     ("%.2f" % 0.00) + " %", 
                                     ("%.2f" % 0.00) + " %"]),
                        ('Away', [predOutcome[1], get_team_name(awayTeamID),
                                     ("%.2f" % 0.00) + " %", 
                                     ("%.2f" % 0.00) + " %"]),
                        ('rid', list(range(0, 4)))
                    ])
    
    outcomeDf = pd.DataFrame.from_dict(outcomeData)
    return outcomeDf, keyPlayersDf
    

def get_prediction_output(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    model = loadModel()
    features = loadBaseFeatures()
    
    result1 = Match.query.with_entities(Match.league_id, Match.season, label('team_api_id',Match.home_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season)
    result2 = Match.query.with_entities(Match.league_id, Match.season, label('team_api_id',Match.away_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season)
    result = result1.union(result2)
    
    teamsDf = pd.DataFrame()
    outcomeHDf = pd.DataFrame()
    keyPlayersHDf = pd.DataFrame()
    
    outcomeADf = pd.DataFrame()
    keyPlayersADf = pd.DataFrame()
    
    for row in result:
        homeTeamID = int(team)
        if not int(row.team_api_id) == int(team):
            awayTeamID = int(row.team_api_id)
            data = {'teamsID': get_team_shortname(awayTeamID)}
            teamsDf = teamsDf.append(data, ignore_index=True)
    
            #If team is playing Home game
            outcome, outcome_prob, matchFeatures = getPrediction(model, features, 
                                          homeTeamID=homeTeamID, awayTeamID=awayTeamID,
                                          season=season)
            
            
            oDf, kpDf = get_prediction_detail(outcome, outcome_prob, matchFeatures, homeTeamID, awayTeamID)
            kpDf['Opponent'] = get_team_shortname(awayTeamID)
            oDf['Opponent'] = get_team_shortname(awayTeamID)
            
            if keyPlayersHDf.shape[0]==0:
                keyPlayersHDf = kpDf
            else:
                frames = [keyPlayersHDf, kpDf]
                keyPlayersHDf = pd.concat(frames)
            
            if outcomeHDf.shape[0]==0:
                outcomeHDf = oDf
            else:
                frames = [outcomeHDf, oDf]
                outcomeHDf = pd.concat(frames)
                
            #If team is playing Away game
            outcome, outcome_prob, matchFeatures = getPrediction(model, features, 
                                          homeTeamID=awayTeamID, awayTeamID=homeTeamID,
                                          season=season)
            
            oDf, kpDf = get_prediction_detail(outcome, outcome_prob, matchFeatures, awayTeamID, homeTeamID)
            kpDf['Opponent'] = get_team_shortname(awayTeamID)
            oDf['Opponent'] = get_team_shortname(awayTeamID)
            
            if keyPlayersADf.shape[0]==0:
                keyPlayersADf = kpDf
            else:
                frames = [keyPlayersADf, kpDf]
                keyPlayersADf = pd.concat(frames)
            
            if outcomeADf.shape[0]==0:
                outcomeADf = oDf
            else:
                frames = [outcomeADf, oDf]
                outcomeADf = pd.concat(frames)
    
    if (keyPlayersADf.shape[0]>0):
        keyPlayersADf['HomePlayerId'] = keyPlayersADf['HomePlayerId'].astype(np.int64)
        keyPlayersADf['AwayPlayerId'] = keyPlayersADf['AwayPlayerId'].astype(np.int64)

    if (keyPlayersADf.shape[0]>0):
        keyPlayersHDf['HomePlayerId'] = keyPlayersHDf['HomePlayerId'].astype(np.int64)
        keyPlayersHDf['AwayPlayerId'] = keyPlayersHDf['AwayPlayerId'].astype(np.int64)

    
    return teamsDf, outcomeHDf, keyPlayersHDf, outcomeADf, keyPlayersADf
    

def get_leagues():
    result = League.query.with_entities(League.name, League.id).distinct()
    df = pd.DataFrame()
    for row in result:
        data = {'League': row.name,'LeagueId': str(row.id), 'Lid': int(row.id)}
        df = df.append(data, ignore_index=True)
    
    
    df = df.sort_values(by='Lid', ascending=True)
    df = df.set_index('Lid')
    lid = League.query.with_entities(League.id).order_by(League.id).first()
    lid = str(lid).replace(',','').replace('(','').replace(')','')
    return df, lid

def get_seasons(league):
    result = Match.query.with_entities(Match.league_id, Match.season).filter(Match.league_id == int(league)).distinct()
    df = pd.DataFrame()
    for row in result:
        data = {'LeagueId':'L' + str(row.league_id), 'Season': row.season, 'SeasonId': str(row.season).replace('/','-')}
        df = df.append(data, ignore_index=True)
    if df.shape[0]>0:
        df = df.sort_values(by='Season', ascending=False)
    return df

def get_league_details(league):
    result = db.session.query(label('Country',Country.name), League.name, League.Founded, League.Confederation, League.Number_of_teams, League.Relegation_to, \
                              League.Current_champions, League.Most_championships, League.TV_partners, League.Domestic_cup, \
                              League.International_cup, League.Infomation_source, League.Status, label('LeagueId',League.id)  \
                              ).join(League, Country.id == League.country_id).filter(League.id == int(league))
    
    df = pd.DataFrame()
    for row in result:
        data = {'LeagueId':'L' + str(row.LeagueId), 'Country': row.Country, 'Name': row.name, 'Founded': str(row.Founded), 'Confederation': row.Confederation, \
                'Number of Teams': str(row.Number_of_teams), 'Relegation To': row.Relegation_to, 'Current Champions': row.Current_champions, \
                'Most Championships': row.Most_championships, 'TV Partners': row.TV_partners, 'Domestic Cup': row.Domestic_cup, \
                'International Cup': row.International_cup, 'Infomation Source': row.Infomation_source, 'Status': row.Status}
        df = df.append(data, ignore_index=True)
    
    return df

def get_league_teams(league):
    result1 = Match.query.with_entities(Match.league_id, Match.season, label('team_api_id',Match.home_team_api_id)).filter(Match.league_id == int(league))
    result2 = Match.query.with_entities(Match.league_id, Match.season, label('team_api_id',Match.away_team_api_id)).filter(Match.league_id == int(league))
    result = result1.union(result2)
    
    df = pd.DataFrame()
    for row in result:
        teams = db.session.query(Team).filter(Team.team_api_id == row.team_api_id).all()
        for team in teams:
            data = {'LeagueId':'L' + str(row.league_id), 'Season':row.season, 'Team Id': team.team_short_name, \
                    'Team Name': team.team_long_name, 'TeamApiId': str(row.team_api_id)
                    }
            df = df.append(data, ignore_index=True)
    if df.shape[0]>0:
        df = df.sort_values(by='Season', ascending=False)
    return df

def get_team_details(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    teamName = ''
    playersDf = pd.DataFrame(columns=['Name','Birth Day','Height','Weight', 'playerID'])
    
    if int(league) == 0:
        return playersDf, teamName
    
    result1 = Match.query.with_entities(label('team_api_id',Match.home_team_api_id), \
                                        label('player1',Match.home_player_1), \
                                        label('player2',Match.home_player_2), \
                                        label('player3',Match.home_player_3), \
                                        label('player4',Match.home_player_4), \
                                        label('player5',Match.home_player_5), \
                                        label('player6',Match.home_player_6), \
                                        label('player7',Match.home_player_7), \
                                        label('player8',Match.home_player_8), \
                                        label('player9',Match.home_player_9), \
                                        label('player10',Match.home_player_10), \
                                        label('player11',Match.home_player_11)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(team))
    
    result2 = Match.query.with_entities(label('team_api_id',Match.away_team_api_id), \
									  label('player1',Match.away_player_1), \
									  label('player2',Match.away_player_2), \
                                       label('player3',Match.away_player_3), \
                                       label('player4',Match.away_player_4), \
                                       label('player5',Match.away_player_5), \
                                       label('player6',Match.away_player_6), \
                                       label('player7',Match.away_player_7), \
                                       label('player8',Match.away_player_8), \
                                       label('player9',Match.away_player_9), \
                                       label('player10',Match.away_player_10), \
                                       label('player11',Match.away_player_11)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(team))
    
    result = result1.union(result2)
    
    df = pd.DataFrame()
    for row in result:
        data = {'TeamID': row.team_api_id, \
                'player1': row.player1, \
                'player2': row.player2, \
                'player3': row.player3, \
                'player4': row.player4, \
                'player5': row.player5, \
                'player6': row.player6, \
                'player7': row.player7, \
                'player8': row.player8, \
                'player9': row.player9, \
                'player10': row.player10, \
                'player11': row.player11
                }
        df = df.append(data, ignore_index=True)
    
    df = pd.melt(df, id_vars=['TeamID'], var_name='playerCol', value_name="playerId")
    df.drop('playerCol', axis=1, inplace=True)
    df = df.drop_duplicates()
    playerList = df['playerId'].tolist()

    players = Player.query.filter(Player.player_api_id.in_(playerList)).all()
    #playerDetails = Player_Attributes.query.filter(Player_Attributes.player_api_id.in_(playerList)).all()
    
    #Get players
    playersDf = pd.DataFrame()
    for row in players:
        bday = pd.to_datetime(row.birthday)
        bday = str(bday.month) + '/' + str(bday.day) + '/' + str(bday.year)
        data = {'Name': row.player_name, 'Birth Day': bday, \
                'Height': ("%.2f" % row.height), 'Weight': ("%.2f" % row.weight), 'playerID': str(row.player_api_id)
                }
        playersDf = playersDf.append(data, ignore_index=True)

    if (playersDf.shape[0] > 0):
        cols = ['Name','Birth Day','Height','Weight', 'playerID']
        playersDf = playersDf[cols]
        
    #Get Player latest stats
#    playerDetailsDf = pd.DataFrame()
#    for row in playerDetails:
#        lastStatsDt = pd.to_datetime(row.date)
#        lsd = str(lastStatsDt.month) + '/' + str(lastStatsDt.day) + '/' + str(lastStatsDt.year)
#        
#        data = {'playerID': str(row.player_api_id), 'lastStatsDt': lastStatsDt, 'lsd': lsd, 'Overall Rating': row.overall_rating, 
#                'Potential': row.potential, 'Preferred Foot': str(row.preferred_foot).title(),
#                'Attacking Work Rate': str(row.attacking_work_rate).title(),
#                'Defensive Work Rate': str(row.defensive_work_rate).title(),
#                'Crossing': row.crossing, 'Finishing Rate': row.finishing, 'Heading Accuracy': row.heading_accuracy, 
#                'Short Passing': row.short_passing, 'Volleys': row.volleys, 'Dribbling Rate': row.dribbling, 
#                'Curve': row.curve, 'Free Kick Accuracy': row.free_kick_accuracy, 'Long Passing': row.long_passing, 
#                'Ball Control': row.ball_control, 'Acceleration': row.acceleration, 'Sprint Speed': row.sprint_speed,
#                'Agility': row.agility, 'Reactions': row.reactions, 'Balance': row.balance, 
#                'Shot Power': row.shot_power, 'Jumping': row.jumping, 'Stamina': row.stamina, 
#                'Strength': row.strength, 'Long Shots': row.long_shots, 'Aggression': row.aggression, 
#                'Interception': row.interceptions, 'Vision': row.vision, 'Positioning': row.positioning,
#                'Penalties': row.penalties, 'Marking': row.marking, 'Standing Tackle': row.standing_tackle, 
#                'Sliding Tackle': row.sliding_tackle, 'Goalkeeping Driving': row.gk_diving, 
#                'Goalkeeping Handling': row.gk_handling, 'Goalkeeping Kicking': row.gk_kicking, 
#                'Goalkeeping Positioning': row.gk_positioning, 'Goalkeeping Reflexes': row.gk_reflexes
#                }
#        playerDetailsDf = playerDetailsDf.append(data, ignore_index=True)
#
# 
#    # groupby first two columns, then get the maximum value in the third column
#    idx = playerDetailsDf.groupby(['playerID'])['lastStatsDt'].transform(max) == playerDetailsDf['lastStatsDt']
#    
#    # use the index to fetch correct rows in DataFrame
#    playerLatest = playerDetailsDf[idx]
    
    teamName = get_team_name(team)
    
    return playersDf, teamName


def get_player_details(player):
    player = int(player)
    playerDetails = Player_Attributes.query.filter(Player_Attributes.player_api_id == player).all()
    #Get Player latest stats
    playerDetailsDf = pd.DataFrame()
    graphDf = pd.DataFrame()
    for row in playerDetails:
        lastStatsDt = pd.to_datetime(row.date)
        lsd = str(lastStatsDt.month) + '/' + str(lastStatsDt.day) + '/' + str(lastStatsDt.year)
        
        data = {'playerID': str(row.player_api_id), 'lastStatsDt': lastStatsDt, 'Rating As Of': lsd, 'Overall Rating': row.overall_rating, 
                'Potential': row.potential, 'Preferred Foot': str(row.preferred_foot).title(),
                'Attacking Work Rate': str(row.attacking_work_rate).title(),
                'Defensive Work Rate': str(row.defensive_work_rate).title(),
                'Crossing': row.crossing, 'Finishing Rate': row.finishing, 'Heading Accuracy': row.heading_accuracy, 
                'Short Passing': row.short_passing, 'Volleys': row.volleys, 'Dribbling Rate': row.dribbling, 
                'Curve': row.curve, 'Free Kick Accuracy': row.free_kick_accuracy, 'Long Passing': row.long_passing, 
                'Ball Control': row.ball_control, 'Acceleration': row.acceleration, 'Sprint Speed': row.sprint_speed,
                'Agility': row.agility, 'Reactions': row.reactions, 'Balance': row.balance, 
                'Shot Power': row.shot_power, 'Jumping': row.jumping, 'Stamina': row.stamina, 
                'Strength': row.strength, 'Long Shots': row.long_shots, 'Aggression': row.aggression, 
                'Interception': row.interceptions, 'Vision': row.vision, 'Positioning': row.positioning,
                'Penalties': row.penalties, 'Marking': row.marking, 'Standing Tackle': row.standing_tackle, 
                'Sliding Tackle': row.sliding_tackle, 'Goalkeeping Driving': row.gk_diving, 
                'Goalkeeping Handling': row.gk_handling, 'Goalkeeping Kicking': row.gk_kicking, 
                'Goalkeeping Positioning': row.gk_positioning, 'Goalkeeping Reflexes': row.gk_reflexes
                }
        playerDetailsDf = playerDetailsDf.append(data, ignore_index=True)
        graphDf = playerDetailsDf.append(data, ignore_index=True)
        
    
    playerDetailsDf = playerDetailsDf.sort_values('lastStatsDt', ascending=False).head(1)
    
    lsd = playerDetailsDf.iloc[0]['Rating As Of']
    
    playerDetailsDf = playerDetailsDf.drop(['Rating As Of', 'lastStatsDt'], axis=1)
   
    playerDetailsDf = pd.melt(playerDetailsDf, id_vars=['playerID'], var_name='Attribute', value_name="Rating")
    
    playerDetailsDf = playerDetailsDf.drop(['playerID'], axis=1)
    
    playerDetailsDf['NewId'] = playerDetailsDf.index + 1
    
    data = {'Attribute': 'Rating As Of', 'Rating': lsd, 'NewId': 0}
    
    playerDetailsDf = playerDetailsDf.append(data, ignore_index=True)
    playerDetailsDf = playerDetailsDf.sort_values('NewId', ascending=True)
    playerDetailsDf = playerDetailsDf.reset_index(drop=True)
    playerDetailsDf = playerDetailsDf.drop(['NewId'], axis=1)
    
    return playerDetailsDf, graphDf

def get_team_winlose(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    df = pd.DataFrame(columns=['Date','Opponent','OpponentId','Outcome','Goals'])
    df1 = pd.DataFrame(columns=['Opponent','Outcome','GoalsMade', 'GoalsGiven','Diff', 'date1'])

    if int(league)==0:
        return df, df1
    
    result1 = Match.query.with_entities(Match.league_id, Match.goal, Match.date,
                                        Match.season, label('team_api_id',Match.home_team_api_id), label('opponent',Match.away_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(team))
    
    result2 = Match.query.with_entities(Match.league_id, Match.goal, Match.date,
                                        Match.season, label('team_api_id',Match.away_team_api_id), label('opponent',Match.home_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(team))
    result = result1.union(result2)
    
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for row in result:
        g = 0
        og = 0
        goalText = row.goal
        if not goalText == None:
            soup = BeautifulSoup(goalText,'xml')
            goals = soup.find_all('goals')
            teamId = soup.find_all('team')
            for i in range(0, len(goals)):
                if int(teamId[i].get_text()) == int(team):
                    g = g + int(goals[i].get_text())
                else:
                    og = og + int(goals[i].get_text())
        outcome = 'D'
        if g > og:
            outcome = 'W'
        if (g < og):
            outcome = 'L'
        
        data = {'Opponent': get_team_name(row.opponent),
                'OpponentId': get_team_shortname(row.opponent),
                'Outcome': outcome,
                'Goals': str(g) + ' - ' + str(og),
                'date1': pd.to_datetime(row.date)
                }
        data1 = {'Opponent': get_team_shortname(row.opponent),
                'Outcome': outcome,
                'GoalsMade': str(g),
                'GoalsGiven': str(og),
                'Diff': (g - og),
                'date1': pd.to_datetime(row.date)
                }
        
        df = df.append(data, ignore_index=True)
        df1 = df1.append(data1, ignore_index=True)
        
    if (df.shape[0] > 0):
        df = df.sort_values(by='date1', ascending=True)
        df1 = df1.sort_values(by='date1', ascending=True)
        df['Date'] = df['date1'] .apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%Y'))
        df = df.set_index('date1')
        df1 = df1.set_index('date1')
        cols = ['Date','Opponent','OpponentId','Outcome','Goals']
        df = df[cols]
    return df, df1

def get_team_shoton(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    result1 = Match.query.with_entities(Match.league_id, Match.shoton, Match.date,
                                        Match.season, label('team_api_id',Match.home_team_api_id), label('opponent',Match.away_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(team))
    
    result2 = Match.query.with_entities(Match.league_id, Match.shoton, Match.date,
                                        Match.season, label('team_api_id',Match.away_team_api_id), label('opponent',Match.home_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(team))
    result = result1.union(result2)
    
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for row in result:
        s = 0
        os = 0
        s1 = 0
        os1 = 0
        shotonText = row.shoton
        if not shotonText == None:
            soup = BeautifulSoup(shotonText,'xml')
            shot = soup.find_all('type')
            teamId = soup.find_all('team')
            for i in range(0, len(shot)):
                if shot[i].get_text() == 'shoton':
                    if len(teamId) > i:
                        if int(teamId[i].get_text()) == int(team):
                            s = s + 1
                        else:
                            os = os + 1
        totalShots = s + os
        if totalShots > 0:
            s1 = (s/totalShots) * 100
            os1 = (os/totalShots) * 100
        
        data = {'Opponent': get_team_name(row.opponent),
                'OpponentId': get_team_shortname(row.opponent),
                'Shot On Goal Ratio': '(' + ("%d" % s) + ')' +  ("%.2f" % s1) + ' : ' + ("%.2f" % os1) + '(' + ("%d" % os) + ')' ,
                'date1': pd.to_datetime(row.date)
                }
        data1 = {'Opponent': get_team_shortname(row.opponent),
                 'SOGT': s,
                 'SOGO': os,
                 'date1': pd.to_datetime(row.date)
                }
        
        df = df.append(data, ignore_index=True)
        df1 = df1.append(data1, ignore_index=True)

    if (df.shape[0] > 0):
        df = df.sort_values(by='date1', ascending=True)
        df1 = df1.sort_values(by='date1', ascending=True)
        df['Date'] = df['date1'] .apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%Y'))
        df = df.set_index('date1')
        df1 = df1.set_index('date1')
        cols = ['Date','Opponent','OpponentId','Shot On Goal Ratio']
        df = df[cols]
    return df, df1

def get_team_shotoff(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    result1 = Match.query.with_entities(Match.league_id, Match.shotoff, Match.date,
                                        Match.season, label('team_api_id',Match.home_team_api_id), label('opponent',Match.away_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(team))
    
    result2 = Match.query.with_entities(Match.league_id, Match.shotoff, Match.date,
                                        Match.season, label('team_api_id',Match.away_team_api_id), label('opponent',Match.home_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(team))
    result = result1.union(result2)
    
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for row in result:
        s = 0
        os = 0
        s1 = 0
        os1 = 0
        shotoffText = row.shotoff
        if not shotoffText == None:
            soup = BeautifulSoup(shotoffText,'xml')
            shot = soup.find_all('type')
            teamId = soup.find_all('team')
            for i in range(0, len(shot)):
                if shot[i].get_text() == 'shotoff':
                    if len(teamId) > i:
                        if int(teamId[i].get_text()) == int(team):
                            s = s + 1
                        else:
                            os = os + 1
        totalShots = s + os
        if totalShots > 0:
            s1 = (s/totalShots) * 100
            os1 = (os/totalShots) * 100
        
        data = {'Opponent': get_team_name(row.opponent),
                'OpponentId': get_team_shortname(row.opponent),
                'Shot On Goal Ratio': '(' + ("%d" % s) + ')' +  ("%.2f" % s1) + ' : ' + ("%.2f" % os1) + '(' + ("%d" % os) + ')' ,
                'date1': pd.to_datetime(row.date)
                }
        data1 = {'Opponent': get_team_shortname(row.opponent),
                 'SOGT': s,
                 'SOGO': os,
                 'date1': pd.to_datetime(row.date)
                }
        
        df = df.append(data, ignore_index=True)
        df1 = df1.append(data1, ignore_index=True)

    if (df.shape[0] > 0):
        df = df.sort_values(by='date1', ascending=True)
        df1 = df1.sort_values(by='date1', ascending=True)
        df['Date'] = df['date1'] .apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%Y'))
        df = df.set_index('date1')
        df1 = df1.set_index('date1')
        cols = ['Date','Opponent','OpponentId','Shot On Goal Ratio']
        df = df[cols]
    return df, df1


def get_team_foulcommits(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    result1 = Match.query.with_entities(Match.league_id, Match.foulcommit, Match.date,
                                        Match.season, label('team_api_id',Match.home_team_api_id), label('opponent',Match.away_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(team))
    
    result2 = Match.query.with_entities(Match.league_id, Match.foulcommit, Match.date,
                                        Match.season, label('team_api_id',Match.away_team_api_id), label('opponent',Match.home_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(team))
    result = result1.union(result2)
    
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for row in result:
        s = 0
        os = 0
        foulcommitText = row.foulcommit
        if not foulcommitText == None:
            soup = BeautifulSoup(foulcommitText,'xml')
            foul = soup.find_all('type')
            teamId = soup.find_all('team')
            for i in range(0, len(foul)):
                if foul[i].get_text() == 'foulcommit':
                    if len(teamId) > i:
                        if int(teamId[i].get_text()) == int(team):
                            s = s + 1
                        else:
                            os = os + 1
        totalFouls = s + os
        if totalFouls > 0:
            s1 = (s/totalFouls) * 100
            os1 = (os/totalFouls) * 100
        else:
            s1 = 0
            os1 = 0
        
        data = {'Opponent': get_team_name(row.opponent),
                'OpponentId': get_team_shortname(row.opponent),
                'Foul Commit Ratio': ("%.2f" % s1)+ ' : ' + ("%.2f" % os1),
                 'date1': pd.to_datetime(row.date)
                }
        data1 = {'Opponent': get_team_shortname(row.opponent),
                 'FoulT': s,
                 'FoulO': os,
                 'date1': pd.to_datetime(row.date)
                }
        
        df = df.append(data, ignore_index=True)
        df1 = df1.append(data1, ignore_index=True)
    
    if (df.shape[0] > 0):
        df = df.sort_values(by='date1', ascending=True)
        df1 = df1.sort_values(by='date1', ascending=True)
        df['Date'] = df['date1'] .apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%Y'))
        df = df.set_index('date1')
        df1 = df1.set_index('date1')
        cols = ['Date','Opponent','OpponentId','Foul Commit Ratio']
        df = df[cols]
    return df, df1


def get_correlation(league, season):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    result1 = Match.query.with_entities(label('team_api_id',Match.away_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season)
    
    result2 = Match.query.with_entities(label('team_api_id',Match.home_team_api_id)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season)
    teams = result = result1.union(result2)

    
    result = Team_Attributes.query.with_entities(Team_Attributes.buildUpPlaySpeed, \
                                                Team_Attributes.buildUpPlayDribbling, \
                                                Team_Attributes.buildUpPlayPassing, \
                                                Team_Attributes.chanceCreationPassing, \
                                                Team_Attributes.chanceCreationCrossing, \
                                                Team_Attributes.chanceCreationShooting, \
                                                Team_Attributes.defencePressure, \
                                                Team_Attributes.defenceAggression, \
                                                Team_Attributes.defenceTeamWidth, \
                                                Team_Attributes.date, Team_Attributes.team_api_id
                                                ).filter(Team_Attributes.team_api_id.in_(teams)).all()
    
    df = pd.DataFrame()
    for row in result:
        lastStatsDt = pd.to_datetime(row.date)
        lsd = str(lastStatsDt.month) + '/' + str(lastStatsDt.day) + '/' + str(lastStatsDt.year)
        
        data = {'Play Speed': row.buildUpPlaySpeed,
                'Dribbling': row.buildUpPlayDribbling,
                'Play Passing': row.buildUpPlayPassing,
                'Creative Passing': row.chanceCreationPassing,
                'Creative Crossing': row.chanceCreationCrossing,
                'Creative Shooting': row.chanceCreationShooting,
                'Defence Pressure': row.defencePressure,
                'Defence Aggression': row.defenceAggression,
                'Defence Team Width': row.defenceTeamWidth,
                'lastStatsDt': lastStatsDt,
                'Lastest Stats Date': lsd,
                'TeamId': row.team_api_id
                }
        df = df.append(data, ignore_index=True)
    
    idx = df.groupby(['TeamId'])['lastStatsDt'].transform(max) == df['lastStatsDt']
    
    # use the index to fetch correct rows in DataFrame
    teamLatestStats = df[idx]
    teamLatestStats.fillna(0)
    
    df = pd.DataFrame()
    for team in teams:
        teamapiId = team.team_api_id
        w = 0
        l = 0
        d = 0
        f = 0
        s = 0
        result1 = Match.query.with_entities(Match.league_id, Match.goal, Match.foulcommit, Match.shoton, Match.date,
                                            Match.season, label('team_api_id',Match.home_team_api_id), label('opponent',Match.away_team_api_id)
                                            ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(teamapiId))
        
        result2 = Match.query.with_entities(Match.league_id, Match.goal, Match.foulcommit, Match.shoton, Match.date,
                                            Match.season, label('team_api_id',Match.away_team_api_id), label('opponent',Match.home_team_api_id)
                                            ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(teamapiId))
        result = result1.union(result2)
    
        for row in result:
            g = 0
            og = 0
            goalText = row.goal
            if not goalText == None:
                soup = BeautifulSoup(goalText,'xml')
                goals = soup.find_all('goals')
                teamId = soup.find_all('team')
                for i in range(0, len(goals)):
                    if len(teamId) > i:
                        if int(teamId[i].get_text()) == int(teamapiId):
                            g = g + int(goals[i].get_text())
                        else:
                            og = og + int(goals[i].get_text())
                        
            foulcommit = row.foulcommit
            if not foulcommit == None:
                soup = BeautifulSoup(foulcommit,'xml')
                foul = soup.find_all('foulcommit')
                teamId = soup.find_all('team')
                for i in range(0, len(foul)):
                    if len(teamId) > i:
                        if int(teamId[i].get_text()) == int(teamapiId):
                            f = f + 1
                        
            shoton = row.shoton
            if not shoton == None:
                soup = BeautifulSoup(shoton,'xml')
                shot = soup.find_all('shoton')
                teamId = soup.find_all('team')
                for i in range(0, len(shot)):
                    if len(teamId) > i:
                        if int(teamId[i].get_text()) == int(teamapiId):
                            s = s + 1

            outcome = 'D'
            if g > og:
                outcome = 'W'
            if (g < og):
                outcome = 'L'
    
            w = w + 1 if outcome == 'W' else w
            l = l + 1 if outcome == 'L' else l
            d = d + 1 if outcome == 'D' else d
        
        data = {'TeamId': teamapiId, 'Wins': w, 'Loses': l, 'Draws': d, 'Foul Commit': f, 'Shot On Goal': s}
        df = df.append(data, ignore_index=True)
        
    df = pd.merge(df, teamLatestStats, on='TeamId', how='inner')
    
    cols = ['Play Speed','Dribbling','Play Passing','Creative Passing','Creative Crossing','Creative Shooting', \
            'Defence Pressure','Defence Aggression','Defence Team Width', 'Wins', 'Loses', 'Draws', 'Foul Commit', 'Shot On Goal']
    df = df[cols]
    
    outcome = ['Wins', 'Draws', 'Loses']
    df = df.corr().filter(outcome).drop(outcome)
    return df
                


def get_teamStats(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    
    result = db.session.query(Team_Attributes).filter(Team_Attributes.team_api_id == int(team)).all()
    
    df = pd.DataFrame()
    df1 = pd.DataFrame()    
    for row in result:
        lastStatsDt = pd.to_datetime(row.date)
        lsd = str(lastStatsDt.month) + '/' + str(lastStatsDt.day) + '/' + str(lastStatsDt.year)
        
        data = {'Play Speed': row.buildUpPlaySpeed,
                'Dribbling': row.buildUpPlayDribbling,
                'Play Passing': row.buildUpPlayPassing,
                'Creative Passing': row.chanceCreationPassing,
                'Creative Crossing': row.chanceCreationCrossing,
                'Creative Shooting': row.chanceCreationShooting,
                'Defence Pressure': row.defencePressure,
                'Defence Aggression': row.defenceAggression,
                'Defence Team Width': row.defenceTeamWidth,
                'lastStatsDt': lastStatsDt,
                'Lastest Stats Date': lsd,
                'TeamId': row.team_api_id
                }
        data1 = {'Play Speed': str(row.buildUpPlaySpeedClass).title(),
                'Dribbling': str(row.buildUpPlayDribblingClass).title(),
                'Play Passing': str(row.buildUpPlayPassingClass).title(),
                'Creative Passing': str(row.chanceCreationPassingClass).title(),
                'Creative Crossing': str(row.chanceCreationCrossingClass).title(),
                'Creative Shooting': str(row.chanceCreationShootingClass).title(),
                'Defence Pressure': str(row.defencePressureClass).title(),
                'Defence Aggression': str(row.defenceAggressionClass).title(),
                'Defence Team Width': str(row.defenceTeamWidthClass).title(),                                          
                'Defence Line': str(row.defenceDefenderLineClass).title(),
                'lastStatsDt': lastStatsDt,
                'Lastest Stats Date': lsd,
                'TeamId': row.team_api_id
                }
        
        
        df = df.append(data, ignore_index=True)
        df1 = df1.append(data1, ignore_index=True)
    
    idx = df.groupby(['TeamId'])['lastStatsDt'].transform(max) == df['lastStatsDt']
    df = df[idx]
    df1 = df1[idx]
    df.fillna('NA')
    df.fillna('NA')
    cols = ['Play Speed','Dribbling','Play Passing','Creative Passing','Creative Crossing','Creative Shooting', \
            'Defence Pressure','Defence Aggression','Defence Team Width', 'TeamId']
    df = df[cols]
    
    cols = ['Play Speed','Dribbling','Play Passing','Creative Passing','Creative Crossing','Creative Shooting', \
            'Defence Pressure','Defence Aggression','Defence Team Width', 'Defence Line', 'TeamId']
    df1 = df1[cols]
    
    df = pd.melt(df, id_vars=['TeamId'], var_name='Attribute', value_name="Rating")
    df1 = pd.melt(df1, id_vars=['TeamId'], var_name='Attribute', value_name="Class")
    
    cols = ['Attribute','Rating']
    df = df[cols]
    
    cols = ['Attribute','Class']
    df1 = df1[cols]
    
    return df, df1


def get_top5(league, season, team):
    league = league.replace('L','')
    season = season.replace('-','/')
    playerList = []
    players = []
    
    goalsDf = pd.DataFrame(columns=['Name', 'Goals', 'Percentage'])
    shotonDf = pd.DataFrame(columns=['Name', 'ShotOn', 'Percentage'])
    foulcommitDf = pd.DataFrame(columns=['Name', 'Fouls', 'Percentage'])
    cardsDf = pd.DataFrame(columns=['Name', 'Cards', 'Percentage'])
    
    if int(league) == 0:
        return goalsDf, shotonDf, foulcommitDf, cardsDf
    
    result1 = Match.query.with_entities(Match.league_id, Match.goal, Match.foulcommit, Match.shoton, Match.card, \
                                        Match.date, Match.season, \
                                        label('team_api_id',Match.home_team_api_id), \
                                        label('player1',Match.home_player_1), \
                                        label('player2',Match.home_player_2), \
                                        label('player3',Match.home_player_3), \
                                        label('player4',Match.home_player_4), \
                                        label('player5',Match.home_player_5), \
                                        label('player6',Match.home_player_6), \
                                        label('player7',Match.home_player_7), \
                                        label('player8',Match.home_player_8), \
                                        label('player9',Match.home_player_9), \
                                        label('player10',Match.home_player_10), \
                                        label('player11',Match.home_player_11)
                                        ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.home_team_api_id == int(team))
    
    result2 = Match.query.with_entities(Match.league_id, Match.goal, Match.foulcommit, Match.shoton, Match.card, \
                                        Match.date, Match.season, \
                                        label('team_api_id',Match.away_team_api_id), \
									   label('player1',Match.away_player_1), \
									   label('player2',Match.away_player_2), \
                                        label('player3',Match.away_player_3), \
                                        label('player4',Match.away_player_4), \
                                        label('player5',Match.away_player_5), \
                                        label('player6',Match.away_player_6), \
                                        label('player7',Match.away_player_7), \
                                        label('player8',Match.away_player_8), \
                                        label('player9',Match.away_player_9), \
                                        label('player10',Match.away_player_10), \
                                        label('player11',Match.away_player_11)
                                       ).filter(Match.league_id == int(league)).filter(Match.season == season).filter(Match.away_team_api_id == int(team))
    result = result1.union(result2)

    df = pd.DataFrame()
    for row in result:
        data = {'TeamID': row.team_api_id, \
                'player1': ("%d" % 0 if row.player1==None else row.player1), \
                'player2': ("%d" % 0 if row.player2==None else row.player2), \
                'player3': ("%d" % 0 if row.player3==None else row.player3), \
                'player4': ("%d" % 0 if row.player4==None else row.player4), \
                'player5': ("%d" % 0 if row.player5==None else row.player5), \
                'player6': ("%d" % 0 if row.player6==None else row.player6), \
                'player7': ("%d" % 0 if row.player7==None else row.player7), \
                'player8': ("%d" % 0 if row.player8==None else row.player8), \
                'player9': ("%d" % 0 if row.player9==None else row.player9), \
                'player10': ("%d" % 0 if row.player10==None else row.player10), \
                'player11': ("%d" % 0 if row.player11==None else row.player11)
                }
        df = df.append(data, ignore_index=True)

    if (df.shape[0] > 0):
        df = pd.melt(df, id_vars=['TeamID'], var_name='playerCol', value_name="playerId")
        df.drop('playerCol', axis=1, inplace=True)
        df = df.drop_duplicates()
        df['playerId'] = df['playerId'].astype(np.int64)
        
        playerList = df['playerId'].tolist()
    
        #Goals
        emptySet = [0] * len(playerList)
        playerGoals = list(zip(playerList,emptySet))
        playerShoton = list(zip(playerList,emptySet))
        playerFouls = list(zip(playerList,emptySet))
        playerCards = list(zip(playerList,emptySet))    

    for row in result:
        goalText = row.goal
        if not goalText == None:
            soup = BeautifulSoup(goalText,'xml')
            goals = soup.find_all('goals')
            teamId = soup.find_all('team')
            players = soup.find_all('player1')
            for i in range(0, len(goals)):
                if len(teamId) > i:
                    if int(teamId[i].get_text()) == int(team):
                        if len(players) > i:
                            player = int(players[i].get_text())
                            playerGoals = [(k,v) if (k != player) else (player, v + 1) for (k, v) in playerGoals]
                    
        foulcommit = row.foulcommit
        if not foulcommit == None:
            soup = BeautifulSoup(foulcommit,'xml')
            foul = soup.find_all('foulcommit')
            teamId = soup.find_all('team')
            players = soup.find_all('player1')
            for i in range(0, len(foul)):
                if len(teamId) > i:
                    if int(teamId[i].get_text()) == int(team):
                        if len(players) > i:
                            player = int(players[i].get_text())
                            playerFouls = [(k,v) if (k != player) else (player, v + 1) for (k, v) in playerFouls]

                    
        shoton = row.shoton
        if not shoton == None:
            soup = BeautifulSoup(shoton,'xml')
            shot = soup.find_all('shoton')
            teamId = soup.find_all('team')
            players = soup.find_all('player1')
            for i in range(0, len(shot)):
                if len(teamId) > i:
                    if int(teamId[i].get_text()) == int(team):
                        if len(players) > i:
                            player = int(players[i].get_text())
                            playerShoton = [(k,v) if (k != player) else (player, v + 1) for (k, v) in playerShoton]
                            
        cards = row.card
        if not shoton == None:
            soup = BeautifulSoup(cards,'xml')
            card = soup.find_all('type')
            teamId = soup.find_all('team')
            players = soup.find_all('player1')
            for i in range(0, len(card)):
                if len(teamId) > i:
                    if int(teamId[i].get_text()) == int(team):
                        if len(players) > i:
                            player = int(players[i].get_text())
                            playerCards = [(k,v) if (k != player) else (player, v + 1) for (k, v) in playerCards]
    
    
    if (len(playerGoals) > 0):   
        goalsDf = pd.DataFrame(playerGoals)
        goalsDf.columns = ['playerId','Goals']
    
    if (len(playerFouls) > 0):
        foulcommitDf = pd.DataFrame(playerFouls)
        foulcommitDf.columns = ['playerId','Fouls']

    if (len(playerShoton) > 0):
        shotonDf = pd.DataFrame(playerShoton)
        shotonDf.columns = ['playerId','ShotOn']

    if (len(playerCards) > 0):
        cardsDf = pd.DataFrame(playerCards)
        cardsDf.columns = ['playerId','Cards']    
    
    if (len(playerList) > 0):
        players = Player.query.filter(Player.player_api_id.in_(playerList)).all()
   
    #Get players
    playersDf = pd.DataFrame()
    for row in players:
        data = {'Name': row.player_name, 'playerId': str(row.player_api_id)}
        playersDf = playersDf.append(data, ignore_index=True)

    if (playersDf.shape[0] > 0):
        cols = ['Name', 'playerId']
        playersDf = playersDf[cols]
        playersDf['playerId'] = playersDf['playerId'].astype(np.int64)
        
    goalsDf = pd.merge(goalsDf, playersDf, on='playerId', how='inner')
    total = goalsDf['Goals'].sum()
    goalsDf = goalsDf.sort_values('Goals', ascending=False).head(5)[['Name', 'Goals']]
    goalsDf.reset_index(drop=True)
    top5 = goalsDf['Goals'].sum()
    balance = total - top5
    data = {'Name': 'Others', 'Goals': balance}
    goalsDf = goalsDf.append(data, ignore_index=True)
    goalsDf['Percentage'] = (goalsDf['Goals'] * (100 / (1 if total==0 else total)))
    
    shotonDf = pd.merge(shotonDf, playersDf, on='playerId', how='inner')
    total = shotonDf['ShotOn'].sum()
    shotonDf = shotonDf.sort_values('ShotOn', ascending=False).head(5)[['Name', 'ShotOn']]
    shotonDf.reset_index(drop=True)
    top5 = shotonDf['ShotOn'].sum()
    balance = total - top5
    data = {'Name': 'Others', 'ShotOn': balance}
    shotonDf = shotonDf.append(data, ignore_index=True)
    shotonDf['Percentage'] = (shotonDf['ShotOn'] * (100 / (1 if total==0 else total)))
    
    foulcommitDf = pd.merge(foulcommitDf, playersDf, on='playerId', how='inner')
    total = foulcommitDf['Fouls'].sum()
    foulcommitDf = foulcommitDf.sort_values('Fouls', ascending=False).head(5)[['Name', 'Fouls']]
    foulcommitDf.reset_index(drop=True)
    top5 = foulcommitDf['Fouls'].sum()
    balance = total - top5
    data = {'Name': 'Others', 'Fouls': balance}
    foulcommitDf = foulcommitDf.append(data, ignore_index=True)
    foulcommitDf['Percentage'] = (foulcommitDf['Fouls'] * (100 / (1 if total==0 else total)))
    
    cardsDf = pd.merge(cardsDf, playersDf, on='playerId', how='inner')
    total = cardsDf['Cards'].sum()
    cardsDf = cardsDf.sort_values('Cards', ascending=False).head(5)[['Name', 'Cards']]
    cardsDf.reset_index(drop=True)
    top5 = cardsDf['Cards'].sum()
    balance = total - top5
    data = {'Name': 'Others', 'Cards': balance}
    cardsDf = cardsDf.append(data, ignore_index=True)
    cardsDf['Percentage'] = (cardsDf['Cards'] * (100 / (1 if total==0 else total)))

    return goalsDf, shotonDf, foulcommitDf, cardsDf
    