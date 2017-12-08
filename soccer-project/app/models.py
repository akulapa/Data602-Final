# -*- coding: utf-8 -*-
from app import db

class Team(db.Model):
    
    __tablename__ = 'Team'
    
    id = db.Column(db.Integer, primary_key=True)
    team_api_id = db.Column(db.Integer)
    team_fifa_api_id = db.Column(db.Integer)
    team_long_name = db.Column(db.String(200))
    team_short_name = db.Column(db.String(200))
    
    def __repr__(self):
        return '<Team %r>' % (self.id)

class Player(db.Model):
    __tablename__ = 'Player'
    
    id = db.Column(db.Integer, primary_key=True)
    player_api_id = db.Column(db.Integer)
    player_name = db.Column(db.String(200))
    player_fifa_api_id = db.Column(db.Integer)
    birthday = db.Column(db.String(200))
    height = db.Column(db.Integer)
    weight = db.Column(db.Integer)

    def __repr__(self):
        return '<Player %r>' % (self.id)

class Country(db.Model):
    
    __tablename__ = 'Country'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))

    def __repr__(self):
        return '<Country %r>' % (self.id)

class League(db.Model):
    
    __tablename__ = 'League'

    id = db.Column(db.Integer, primary_key=True)
    country_id = db.Column(db.Integer)
    name = db.Column(db.String(200))
    Founded = db.Column(db.String(200))
    Confederation = db.Column(db.String(200))
    Number_of_teams = db.Column(db.Integer)
    Relegation_to = db.Column(db.String(200))
    Current_champions = db.Column(db.String(200))
    Most_championships = db.Column(db.String(200))
    TV_partners = db.Column(db.String(200))
    Domestic_cup = db.Column(db.String(200))
    International_cup = db.Column(db.String(200))
    Infomation_source = db.Column(db.String(200))
    Status = db.Column(db.String(200))
    

    def __repr__(self):
        return '<League %r>' % (self.id)

class Team_Attributes(db.Model):
    
    __tablename__ = 'Team_Attributes'
        
    id = db.Column(db.Integer, primary_key=True)
    team_fifa_api_id = db.Column(db.Integer)
    team_api_id = db.Column(db.Integer)
    date = db.Column(db.String(200))
    buildUpPlaySpeed = db.Column(db.Integer)
    buildUpPlaySpeedClass = db.Column(db.String(200))
    buildUpPlayDribbling = db.Column(db.Integer)
    buildUpPlayDribblingClass = db.Column(db.String(200))
    buildUpPlayPassing = db.Column(db.Integer)
    buildUpPlayPassingClass = db.Column(db.String(200))
    buildUpPlayPositioningClass = db.Column(db.String(200))
    chanceCreationPassing = db.Column(db.Integer)
    chanceCreationPassingClass = db.Column(db.String(200))
    chanceCreationCrossing = db.Column(db.Integer)
    chanceCreationCrossingClass = db.Column(db.String(200))
    chanceCreationShooting = db.Column(db.Integer)
    chanceCreationShootingClass = db.Column(db.String(200))
    chanceCreationPositioningClass = db.Column(db.String(200))
    defencePressure = db.Column(db.Integer)
    defencePressureClass = db.Column(db.String(200))
    defenceAggression = db.Column(db.Integer)
    defenceAggressionClass = db.Column(db.String(200))
    defenceTeamWidth = db.Column(db.Integer)
    defenceTeamWidthClass = db.Column(db.String(200))
    defenceDefenderLineClass = db.Column(db.String(200))

    def __repr__(self):
        return '<Team_Attributes %r>' % (self.id)


class Player_Attributes(db.Model):
    
    __tablename__ = 'Player_Attributes'

    id = db.Column(db.Integer, primary_key=True)
    player_fifa_api_id = db.Column(db.Integer)
    player_api_id = db.Column(db.Integer)
    date = db.Column(db.String(200))
    overall_rating = db.Column(db.Integer)
    potential = db.Column(db.Integer)
    preferred_foot = db.Column(db.String(200))
    attacking_work_rate = db.Column(db.String(200))
    defensive_work_rate = db.Column(db.String(200))
    crossing = db.Column(db.Integer)
    finishing = db.Column(db.Integer)
    heading_accuracy = db.Column(db.Integer)
    short_passing = db.Column(db.Integer)
    volleys = db.Column(db.Integer)
    dribbling = db.Column(db.Integer)
    curve = db.Column(db.Integer)
    free_kick_accuracy = db.Column(db.Integer)
    long_passing = db.Column(db.Integer)
    ball_control = db.Column(db.Integer)
    acceleration = db.Column(db.Integer)
    sprint_speed = db.Column(db.Integer)
    agility = db.Column(db.Integer)
    reactions = db.Column(db.Integer)
    balance = db.Column(db.Integer)
    shot_power = db.Column(db.Integer)
    jumping = db.Column(db.Integer)
    stamina = db.Column(db.Integer)
    strength = db.Column(db.Integer)
    long_shots = db.Column(db.Integer)
    aggression = db.Column(db.Integer)
    interceptions = db.Column(db.Integer)
    positioning = db.Column(db.Integer)
    vision = db.Column(db.Integer)
    penalties = db.Column(db.Integer)
    marking = db.Column(db.Integer)
    standing_tackle = db.Column(db.Integer)
    sliding_tackle = db.Column(db.Integer)
    gk_diving = db.Column(db.Integer)
    gk_handling = db.Column(db.Integer)
    gk_kicking = db.Column(db.Integer)
    gk_positioning = db.Column(db.Integer)
    gk_reflexes = db.Column(db.Integer)

    def __repr__(self):
        return '<Team_Attributes %r>' % (self.id)


class Match(db.Model):
    
    __tablename__ = 'Match'
   
    id = db.Column(db.Integer, primary_key=True)
    country_id = db.Column(db.Integer)
    league_id = db.Column(db.Integer)
    season = db.Column(db.String(200))
    stage = db.Column(db.Integer)
    date = db.Column(db.String(200))
    match_api_id = db.Column(db.Integer)
    home_team_api_id = db.Column(db.Integer)
    away_team_api_id = db.Column(db.Integer)
    home_team_goal = db.Column(db.Integer)
    away_team_goal = db.Column(db.Integer)
    home_player_X1 = db.Column(db.Integer)
    home_player_X2 = db.Column(db.Integer)
    home_player_X3 = db.Column(db.Integer)
    home_player_X4 = db.Column(db.Integer)
    home_player_X5 = db.Column(db.Integer)
    home_player_X6 = db.Column(db.Integer)
    home_player_X7 = db.Column(db.Integer)
    home_player_X8 = db.Column(db.Integer)
    home_player_X9 = db.Column(db.Integer)
    home_player_X10 = db.Column(db.Integer)
    home_player_X11 = db.Column(db.Integer)
    away_player_X1 = db.Column(db.Integer)
    away_player_X2 = db.Column(db.Integer)
    away_player_X3 = db.Column(db.Integer)
    away_player_X4 = db.Column(db.Integer)
    away_player_X5 = db.Column(db.Integer)
    away_player_X6 = db.Column(db.Integer)
    away_player_X7 = db.Column(db.Integer)
    away_player_X8 = db.Column(db.Integer)
    away_player_X9 = db.Column(db.Integer)
    away_player_X10 = db.Column(db.Integer)
    away_player_X11 = db.Column(db.Integer)
    home_player_Y1 = db.Column(db.Integer)
    home_player_Y2 = db.Column(db.Integer)
    home_player_Y3 = db.Column(db.Integer)
    home_player_Y4 = db.Column(db.Integer)
    home_player_Y5 = db.Column(db.Integer)
    home_player_Y6 = db.Column(db.Integer)
    home_player_Y7 = db.Column(db.Integer)
    home_player_Y8 = db.Column(db.Integer)
    home_player_Y9 = db.Column(db.Integer)
    home_player_Y10 = db.Column(db.Integer)
    home_player_Y11 = db.Column(db.Integer)
    away_player_Y1 = db.Column(db.Integer)
    away_player_Y2 = db.Column(db.Integer)
    away_player_Y3 = db.Column(db.Integer)
    away_player_Y4 = db.Column(db.Integer)
    away_player_Y5 = db.Column(db.Integer)
    away_player_Y6 = db.Column(db.Integer)
    away_player_Y7 = db.Column(db.Integer)
    away_player_Y8 = db.Column(db.Integer)
    away_player_Y9 = db.Column(db.Integer)
    away_player_Y10 = db.Column(db.Integer)
    away_player_Y11 = db.Column(db.Integer)
    home_player_1 = db.Column(db.Integer)
    home_player_2 = db.Column(db.Integer)
    home_player_3 = db.Column(db.Integer)
    home_player_4 = db.Column(db.Integer)
    home_player_5 = db.Column(db.Integer)
    home_player_6 = db.Column(db.Integer)
    home_player_7 = db.Column(db.Integer)
    home_player_8 = db.Column(db.Integer)
    home_player_9 = db.Column(db.Integer)
    home_player_10 = db.Column(db.Integer)
    home_player_11 = db.Column(db.Integer)
    away_player_1 = db.Column(db.Integer)
    away_player_2 = db.Column(db.Integer)
    away_player_3 = db.Column(db.Integer)
    away_player_4 = db.Column(db.Integer)
    away_player_5 = db.Column(db.Integer)
    away_player_6 = db.Column(db.Integer)
    away_player_7 = db.Column(db.Integer)
    away_player_8 = db.Column(db.Integer)
    away_player_9 = db.Column(db.Integer)
    away_player_10 = db.Column(db.Integer)
    away_player_11 = db.Column(db.Integer)
    goal = db.Column(db.String(200))
    shoton = db.Column(db.String(200))
    shotoff = db.Column(db.String(200))
    foulcommit = db.Column(db.String(200))
    card = db.Column(db.String(200))
    cross = db.Column(db.String(200))
    corner = db.Column(db.String(200))
    possession = db.Column(db.String(200))
    B365H = db.Column(db.Float(10))
    B365D = db.Column(db.Float(10))
    B365A = db.Column(db.Float(10))
    BWH = db.Column(db.Float(10))
    BWD = db.Column(db.Float(10))
    BWA = db.Column(db.Float(10))
    IWH = db.Column(db.Float(10))
    IWD = db.Column(db.Float(10))
    IWA = db.Column(db.Float(10))
    LBH = db.Column(db.Float(10))
    LBD = db.Column(db.Float(10))
    LBA = db.Column(db.Float(10))
    PSH = db.Column(db.Float(10))
    PSD = db.Column(db.Float(10))
    PSA = db.Column(db.Float(10))
    WHH = db.Column(db.Float(10))
    WHD = db.Column(db.Float(10))
    WHA = db.Column(db.Float(10))
    SJH = db.Column(db.Float(10))
    SJD = db.Column(db.Float(10))
    SJA = db.Column(db.Float(10))
    VCH = db.Column(db.Float(10))
    VCD = db.Column(db.Float(10))
    VCA = db.Column(db.Float(10))
    GBH = db.Column(db.Float(10))
    GBD = db.Column(db.Float(10))
    GBA = db.Column(db.Float(10))
    BSH = db.Column(db.Float(10))
    BSD = db.Column(db.Float(10))
    BSA = db.Column(db.Float(10))
    
    def __repr__(self):
        return '<Team_Attributes %r>' % (self.id)