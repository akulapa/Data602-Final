#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 2017

@authors: Ilya Kats, Nnaemezue Obi-Eyisi Pavan Akula
"""

from flask import render_template, make_response, flash, request
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.dates as mdates #import DateFormatter
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
from scipy.stats import norm
from app import app


from .dbfunctions import get_teams, get_leagues, get_league_teams, get_seasons, get_league_details, get_team_details, get_player_details, \
get_team_winlose, get_team_name, get_team_shoton, get_team_shotoff, get_team_foulcommits, get_correlation, get_teamStats, get_top5, get_playername, \
get_prediction_output

from .model_test import generate_modelpickle
from .model_training import generate_featurepickle

def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter

def pd_to_html_num_formatters(df):
    keys=[]
    values=[]
    float_format2 = lambda x: '{:,.2f}'.format(x)
    float_format4 = lambda x: '{:,.4f}'.format(x)
    num_format = lambda x: '{:,}'.format(x)
    for (column, dtype) in df.dtypes.iteritems():
        if (dtype in [np.dtype('int64'), np.dtype('float64')]):
            keys.append(column)
            if (column == 'Daily Returns'):
                values.append(float_format4)
            else:
                values.append(num_format if dtype == np.dtype('int64') else float_format2)
    return(dict(zip(keys, values)))

@app.route('/', methods=['GET', 'POST'])
@app.route('/home/<league>', methods=['GET', 'POST'])
def home(league=None):
    #genpickle = generate_featurepickle()
    #genpickle = generate_modelpickle()

    leaugeTable,lid = get_leagues()
    lid = str(lid)
    if league==None:
        league = lid
        
    leaugeTeams = get_league_teams(league)
    leaugeSeasons = get_seasons(league)
    leaugeDetails = get_league_details(league)
#                           leaugeTeams=leaugeTeams,
#                           leaugeSeasons = leaugeSeasons,
#                           leaugeDetails = leaugeDetails,
    return render_template('home.html', 
                           leaugeTable=leaugeTable, 
                           leaugeSeasons=leaugeSeasons,
                           leaugeDetails=leaugeDetails,
                           leaugeTeams=leaugeTeams,
                           league=league,
                           teamid='',
                           season='',
                           playerId='',
                           home = 'class=active data-toggle=pill',
                           team='class=disabled', 
                           player='class=disabled'
                           )

@app.route('/team', defaults={'league': 'L21518', 'season': '2015-2016', 'teamid': 8634})
@app.route('/team/<league>/<season>/<teamid>', methods=['GET', 'POST'])
def team(league=None,season=None,teamid=None):
    
    #print(league, season, teamid)
    teamsDf, outcomeHDf, keyPlayersHDf, outcomeADf, keyPlayersADf = get_prediction_output(league, season, teamid)
    
    top5g, top5s, top5f, top5c =get_top5(league, season, teamid)
    teamDetails, teamName = get_team_details(league, season, teamid)
    gameoutcome, winloseDf = get_team_winlose(league, season, teamid)
    
    w = len(winloseDf[winloseDf['Diff']>0].count(1))
    l = len(winloseDf[winloseDf['Diff']<0].count(1))
    d = len(winloseDf[winloseDf['Diff']==0].count(1))
    
    g = top5g.head(1)
    mg = g.iloc[0]['Name'] + '(' + str(g.iloc[0]['Goals']) + ')'
    s = top5s.head(1)
    ms = s.iloc[0]['Name'] + '(' + str(s.iloc[0]['ShotOn']) + ')'
    f = top5f.head(1)
    mf = f.iloc[0]['Name'] + '(' + str(f.iloc[0]['Fouls']) + ')'
    c = top5c.head(1)
    mc = c.iloc[0]['Name'] + '(' + str(c.iloc[0]['Cards']) + ')'
    
    statslist = [('Won', w), ('Draw',d), ('Lost', l), ('Most Goals', mg), 
                 ('Most Shots on Goal', ms), ('Most Fouls', mf), ('Most Cards', mc)]
    

    #teamDetails = get_team_details()
    num_formatters = pd_to_html_num_formatters(gameoutcome)
    gameoutcome = [gameoutcome.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    shoton, other = get_team_shoton(league, season, teamid)
    num_formatters = pd_to_html_num_formatters(shoton)
    shoton = [shoton.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    shotoff, other = get_team_shotoff(league, season, teamid)
    num_formatters = pd_to_html_num_formatters(shotoff)
    shotoff = [shotoff.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    foulcommit, other = get_team_foulcommits(league, season, teamid)
    num_formatters = pd_to_html_num_formatters(foulcommit)
    foulcommit = [foulcommit.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    corrDf = get_correlation(league, season)
    num_formatters = pd_to_html_num_formatters(corrDf)
    corrDf = [corrDf.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=True)]
    
    teamStats1, teamStats2 = get_teamStats(league, season, teamid)

    num_formatters = pd_to_html_num_formatters(top5g)
    top5g = [top5g.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    num_formatters = pd_to_html_num_formatters(top5s)
    top5s = [top5s.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    num_formatters = pd_to_html_num_formatters(top5f)
    top5f = [top5f.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]
    
    num_formatters = pd_to_html_num_formatters(top5c)
    top5c = [top5c.to_html(formatters = num_formatters, classes="table table-striped table-bordered table-sm", index=False)]

    #print(teamStats1.dtype)
    #teamStats1 = num_formatters = pd_to_html_num_formatters(teamStats1)
    #teamStats1 = [teamStats1.to_html(formatters = num_formatters, classes="table table-striped table-bordered", index=True)]


    return render_template('team.html', 
                           teamDetails=teamDetails,
                           teamName=teamName,
                           gameoutcome = gameoutcome,
                           shoton=shoton,
                           shotoff=shotoff,
                           foulcommit=foulcommit,
                           corrDf=corrDf,
                           teamStats1=teamStats1,
                           teamStats2=teamStats2,
                           top5g=top5g,
                           top5s=top5s,
                           top5f=top5f,
                           top5c=top5c,
                           statslist=statslist,
                           league=league,
                           teamid=teamid,
                           season=season,
                           playerId='',
                           teamsDf=teamsDf,
                           outcomeADf=outcomeADf,
                           keyPlayersADf=keyPlayersADf,
                           outcomeHDf=outcomeHDf,
                           keyPlayersHDf=keyPlayersHDf,
                           home = '',
                           team='class=active data-toggle=pill',
                           player='class=disabled'
                           )

@app.route('/player/<playerId>/<league>/<season>/<teamid>', methods=['GET', 'POST'])
def player(playerId=None,league=None,season=None,teamid=None):
    
    skills = ['Overall Rating','Potential','Preferred Foot','Attacking Work Rate','Defensive Work Rate','Crossing', 'Finishing Rate', 'Heading Accuracy', 
              'Short Passing', 'Volleys', 'Dribbling Rate','Curve', 'Free Kick Accuracy', 'Long Passing','Ball Control', 'Acceleration', 'Sprint Speed',
              'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots', 'Aggression', 'Interception', 'Vision', 'Positioning',
              'Penalties', 'Marking', 'Standing Tackle', 'Sliding Tackle', 'Goalkeeping']

    nskills = ['Preferred Foot','Attacking Work Rate','Defensive Work Rate','Crossing', 'Heading Accuracy', 
              'Short Passing', 'Volleys', 'Dribbling Rate','Curve', 'Long Passing', 'Sprint Speed',
              'Agility', 'Reactions', 'Balance', 'Jumping', 'Strength', 'Long Shots', 'Interception', 'Vision', 'Positioning',
              'Marking', 'Standing Tackle', 'Sliding Tackle', 'Goalkeeping', 'Rating As Of']
    
    playerDf, otherDf = get_player_details(playerId)
    playerName = get_playername(playerId)
    
    return render_template('player.html',
                           playerId = playerId,
                           playerDf = playerDf,
                           playerName=playerName,
                           otherDf=otherDf,
                           skills=skills,
                           nskills=nskills,
                           league=league,
                           teamid=teamid,
                           season=season,
                           home = '',
                           team='',
                           player='class=active data-toggle=pill'
                           )


@app.route("/graph_display1.png/<playerID>/<playerfeature>/<player>")
def graph_display1(playerID, playerfeature, player):
    
    playerID = int(playerID)
    plt.close('all')
    
    otherDf, playerDf = get_player_details(playerID)
    playerDf = playerDf.dropna()
    #Convert text to date
    playerDf['Date_pd'] = pd.to_datetime(playerDf['Rating As Of'])
    playerDf = playerDf.sort_values(by='Date_pd', ascending=True)
    
    #Convert date to string format
    playerDf['date1'] = playerDf['Date_pd'] .apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%Y'))
    
    skills = ['Preferred Foot','Attacking Work Rate','Defensive Work Rate','Crossing', 'Heading Accuracy', 
              'Short Passing', 'Volleys', 'Dribbling Rate','Curve', 'Long Passing', 'Sprint Speed',
              'Agility', 'Reactions', 'Balance', 'Jumping', 'Strength', 'Long Shots', 'Interception', 'Vision', 'Positioning',
              'Marking', 'Standing Tackle', 'Sliding Tackle', 'Goalkeeping']
    
    if (playerfeature not in skills):
        playerDf['data'] = playerDf.date1.str.cat(playerDf[playerfeature].astype("str"), sep=',')

        resultList = playerDf.data.tolist()
        
        x, y = np.loadtxt(resultList,
                          delimiter=',',
                          unpack=True,
                          converters={0: bytespdate2num('%m/%d/%Y')})

    
    plt.close('all')
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(x, y)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.xlabel('Date')
    plt.ylabel(playerfeature)
    plt.title('Changes Over Time - ' + player)
    fig.autofmt_xdate()
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response

@app.route("/graph_display2.png/<playerID>/<playerfeature>/<player>")
def graph_display2(playerID, playerfeature, player):
    
    playerID = int(playerID)
    plt.close('all')
    
    otherDf, playerDf = get_player_details(playerID)
    playerDf = playerDf.dropna()
    #Convert text to date
    playerDf['Date_pd'] = pd.to_datetime(playerDf['Rating As Of'])
    playerDf = playerDf.sort_values(by='Date_pd', ascending=True)
    
    #Convert date to string format
    playerDf['date1'] = playerDf['Date_pd'] .apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%Y'))
    
    if (playerfeature == 'Goalkeeping'):
        playerDf['data'] = playerDf.date1.astype("str") + ',' + playerDf['Goalkeeping Driving'].astype("str") \
                    + ',' + playerDf['Goalkeeping Handling'].astype("str") + ',' + playerDf['Goalkeeping Kicking'].astype("str") \
                    + ',' + playerDf['Goalkeeping Positioning'].astype("str") + ',' + playerDf['Goalkeeping Reflexes'].astype("str")
    

        resultList = playerDf.data.tolist()
        
        x, y1, y2, y3, y4, y5 = np.loadtxt(resultList,
                          delimiter=',',
                          unpack=True,
                          converters={0: bytespdate2num('%m/%d/%Y')})

    
    plt.close('all')
    fig, ax = plt.subplots()
    ax.grid(True)
       
    ax.plot(x, y1, color='navy', alpha=.50, label='Driving')
    ax.plot(x, y2, color='blue', alpha=.50, label='Handling')
    ax.plot(x, y3, color='cyan', alpha=.50, label='Kicking')
    ax.plot(x, y4, color='green', alpha=.50, label='Positioning')
    ax.plot(x, y5, color='red', alpha=.50, label='Reflexes')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.xlabel('Date')
    plt.ylabel(playerfeature)
    
    fig.autofmt_xdate()
    fig = plt.gcf()
    plt.legend(loc='best', fancybox=True)
    plt.title('Change in Goalkeeping Skill - ' + player)
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
    
    # see text.Text, lines.Line2D, and patches.Rectangle for more info on
    # the settable properties of lines, text, and rectangles
    frame.set_facecolor('0.90')      # set the frame face color to light gray
    plt.setp(ltext, fontsize='x-small')    # the legend text fontsize
    plt.setp(llines, linewidth=1)      # the legend linewidth
    leg.get_frame().set_alpha(0.5)
        
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response

@app.route("/graph_display3.png/<league>/<season>/<teamid>")
def graph_display3(league=None,season=None,teamid=None):
    other, df = get_team_winlose(league, season, teamid)
    teamName = get_team_name(teamid)
    
    labels = df.Opponent.tolist()
    x = np.arange(len(labels))
    y = df.GoalsMade.tolist()
    
    plt.close('all')
    fig, ax = plt.subplots()

    ax.set_xlim([0,len(labels)])
    ax.bar(x, y, color='navy', align='center', edgecolor='white',alpha=.80)
    ax.set_xticklabels(labels)
    ax.set_xticks(x, minor = True)
    plt.xticks(x, labels)
    
    ax.set_xticklabels(labels, rotation=90)
    
    plt.xlabel('Opponent')
    plt.ylabel('Goals')
    plt.title(teamName + ' - Goals Made During The Season ' + season)
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response

@app.route("/graph_display4.png/<league>/<season>/<teamid>")
def graph_display4(league=None,season=None,teamid=None):
    other, df = get_team_shoton(league, season, teamid)
    teamName = get_team_name(teamid)
    
    #df['shoton50T'] = 50.00 - df['SOGT']
    #df['shoton50O'] = 50.00 - df['SOGO']
    
    labels = df.Opponent.tolist()
    x = np.arange(len(labels))
    y1 = df.SOGT.tolist()
    y2 = df.SOGO.tolist()
  
    
    plt.close('all')
    fig, ax = plt.subplots()
    #ax.grid(True)
    ax.set_xlim([0,len(labels)])
    ax.bar(x, y1, color='#d62728', label=teamName)
    ax.bar(x, y2, label='Opponent', bottom=y1)
    #ax.axhline(y=50.00, color='k',linewidth=1,linestyle='-')
    ax.legend()
    ax.set_xticks(x, minor = True)
    plt.xticks(x, labels)
    
    ax.set_xticklabels(labels, rotation=90)

    #plt.xticks(rotation=90)
#    for label in ax.xaxis.get_ticklabels():
#        label.set_rotation(90)
    
    plt.xlabel('Opponent')
    plt.ylabel('Shoton Goal')
    plt.title(teamName + ' - Shot On Goal During The Season ' + season)
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response
    
@app.route("/graph_display5.png/<league>/<season>/<teamid>")
def graph_display5(league=None,season=None,teamid=None):
    other, df = get_team_shotoff(league, season, teamid)
    teamName = get_team_name(teamid)
    
    #df['shotoff50T'] = 50.00 - df['SOGT']
    #df['shotoff50O'] = 50.00 - df['SOGO']
    
    labels = df.Opponent.tolist()
    x = np.arange(len(labels))
    y1 = df.SOGT.tolist()
    y2 = df.SOGO.tolist()
  
    
    plt.close('all')
    fig, ax = plt.subplots()
    #ax.grid(True)
    ax.set_xlim([0,len(labels)])
    ax.bar(x, y1, color='#d62728', label=teamName)
    ax.bar(x, y2, label='Opponent', bottom=y1)
    #ax.axhline(y=50.00, color='k',linewidth=1,linestyle='-')
    ax.legend()
    ax.set_xticks(x, minor = True)
    plt.xticks(x, labels)
    
    ax.set_xticklabels(labels, rotation=90)

    #plt.xticks(rotation=90)
#    for label in ax.xaxis.get_ticklabels():
#        label.set_rotation(90)
    
    plt.xlabel('Opponent')
    plt.ylabel('Shotoff Goal')
    plt.title(teamName + ' - Shot Off Goal During The Season ' + season)
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response

@app.route("/graph_display6.png/<league>/<season>/<teamid>")
def graph_display6(league=None,season=None,teamid=None):
    other, df = get_team_foulcommits(league, season, teamid)
    teamName = get_team_name(teamid)
    
    #df['foul50T'] = 50.00 - df['FoulT']
    #df['foul50O'] = 50.00 - df['FoulO']
    
    labels = df.Opponent.tolist()
    x = np.arange(len(labels))
    y1 = df.FoulT.tolist()
    y2 = df.FoulO.tolist()
  
    
    plt.close('all')
    fig, ax = plt.subplots()
    #ax.grid(True)
    ax.set_xlim([0,len(labels)])
    #ax.bar(x, y1, color='green', label='Opponent', bottom=50.00, alpha=.80)
    #ax.bar(x, y2, color='red', label=teamName, bottom=50.00, alpha=.80)
    #ax.axhline(y=50.00, color='k',linewidth=1,linestyle='-')
    ax.bar(x, y1, color='#d62728', label=teamName)
    ax.bar(x, y2, label='Opponent', bottom=y1)
    ax.legend()
    ax.set_xticks(x, minor = True)
    plt.xticks(x, labels)
    
    ax.set_xticklabels(labels, rotation=90)

    #plt.xticks(rotation=90)
#    for label in ax.xaxis.get_ticklabels():
#        label.set_rotation(90)
    
    plt.xlabel('Opponent')
    plt.ylabel('Fouls')
    plt.title(teamName + ' - Fouls Commited During The Season ' + season)
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response

@app.route("/graph_display7.png/<league>/<season>/<teamid>")
def graph_display7(league=None,season=None,teamid=None):
    other, df = get_team_winlose(league, season, teamid)
    teamName = get_team_name(teamid)

    labels = df.Opponent.tolist()    
    df = df.reset_index()
    #x = np.arange(len(labels))
    #y = df.Diff.tolist()

    x = df.index.values
    y = df['Diff']


    d = df['Diff'] == 0
    w = df['Diff'] > 0
    l = df['Diff'] < 0
    
    plt.close('all')
    fig, ax = plt.subplots()
    #ax.grid(True)
    ax.set_xlim([0,len(labels)])
    ax.bar(x[w], y[w], color='green', align='center', edgecolor='white',alpha=.80,label='Won By Goals')
    ax.bar(x[l], abs(y[l]), color='red', align='center', edgecolor='white',alpha=.80,label='Lost By Goals')
    plt.scatter(x[d], y[d], s=30,c='navy',alpha=.80,label='Draw')
    ax.set_xticklabels(labels)
    ax.set_xticks(x, minor = True)
    plt.xticks(x, labels)
    ax.legend()
    ax.set_xticklabels(labels, rotation=90)
    
    plt.xlabel('Opponent')
    plt.ylabel('Win/Lost By')
    plt.title(teamName + ' - Matches Outcome - Season ' + season)
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig) 
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    plt.clf()
    return response

@app.route("/graph_display8.png/<league>/<season>/<teamid>")
def graph_display8(league=None,season=None,teamid=None):
    #https://pythonprogramming.net/stock-price-correlation-table-python-programming-for-finance/
    plt.close('all')

    corrDf = get_correlation(league, season)

    corrMatrix = corrDf.as_matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    heatmap1 = ax.pcolor(corrDf.values, cmap=plt.cm.RdYlGn)
    for y in range(corrMatrix.shape[0]):
        for x in range(corrMatrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % corrMatrix[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
    fig.colorbar(heatmap1)

    ax.set_xticks(np.arange(corrDf.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(corrDf.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels = corrDf.columns
    row_labels = corrDf.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    #plt.xticks(rotation=45)
    #plt.yticks(rotation=45)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

@app.route("/graph_display9.png/<league>/<season>/<teamid>/<top5>")
def graph_display9(league=None,season=None,teamid=None,top5=None):

    plt.close('all')
    
    top5g, top5s, top5f, top5c = get_top5(league, season, teamid)
    teamName = get_team_name(teamid)
    
    if (top5 == 'Goals'):
        top5list = top5g[top5g['Goals'] > 0 ]
        title = teamName + ' Top 5 Goal Scorers'
    elif (top5 == 'Shots'):
        top5list = top5s[top5s['ShotOn'] > 0 ]
        title = teamName + ' - Top 5 Players - Most Shots On Goal'
    elif (top5 == 'Fouls'):
        top5list = top5f[top5f['Fouls'] > 0 ]
        title = teamName + ' - Top 5 Players - Most Fouls'
    else:
        top5list = top5c[top5c['Cards'] > 0 ]
        title = teamName + ' - Top 5 Players - Most Cards'

    labels = top5list['Name'].tolist()
    sizes = tuple(top5list['Percentage'])
    
    cmap = plt.get_cmap('tab20b')
    colors = cmap(np.linspace(0, 1, len(labels)))
    
    #fig1, ax1 = plt.subplots()
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(111)

    #patches, texts, autotexts = ax.pie(sizes, labels=labels, shadow=False, startangle=90, radius=0.5, colors=colors, labeldistance=1.05, autopct='%1.1f%%')
    ax.pie(sizes, shadow=False, startangle=90, radius=0.5, colors=colors, autopct='%1.2f%%')
    ax.legend(labels, loc = 4, fancybox=True)
    ax.axis('equal')
    plt.title(title)
    #plt.legend(loc='best', fancybox=True)
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
    
    # see text.Text, lines.Line2D, and patches.Rectangle for more info on
    # the settable properties of lines, text, and rectangles
    frame.set_facecolor('0.90')      # set the frame face color to light gray
    plt.setp(ltext, fontsize='x-small')    # the legend text fontsize
    plt.setp(llines, linewidth=1)      # the legend linewidth
    leg.get_frame().set_alpha(0.5)
    
#    for t in texts:
#        t.set_size('x-small')
#    for t in autotexts:
#        t.set_size('x-small')
    fig = plt.gcf()
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response