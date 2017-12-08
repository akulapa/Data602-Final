### Data 602 Final Project: SOCCER DATA ANALYSIS TOOL
### CUNY MSDS Program, Fall 2017
### Pavan Akula, Nnaemezue Obi-eyisi, Ilya Kats


#### Objective

The goal of this proof-of-concept project is to design an application that allows analysis of soccer data. The application should include descriptive and statistical features as well as predictive model.

#### Summary

TBD
Include correlation between attributes and win/losses

#### A Note about Soccer Data

Out of major sports, soccer probably has fewest data points that can be measured objectively. It is a long, low-scoring game where a single mistake can make or break it thus increasing randomness of outcomes. Individual and team skill levels have a great affect on game outcome, but they are also very subjective. This data set uses FIFA (International Federation of Association Football) statistics for team and player attributes that was originally scrapped from official FIFA video game. It is unlikely that we will experience the Moneyball moment in soccer in the near future.

#### User Interface/App Functionality

To start a user picks a league, then a season, and then a team. The user then will be presented with the information relevant to the team's performance during selected season. The following information is presented: 

- Player list including age, height and weight
- List of matches with outcomes
- Goal and foul counts per match
- Shooting comparison per match that includes _On Goal_ shots and _Off Goal_ shots
- Top players in the following categories: Goals, Shots, Fouls and Cards. 
- Correlation between team attributes and match outcome (attributes cover team's play, shooting, defense, and attack)
- Prediction for match outcomes against other teams in the league (see _Model_ section below for more information about predictive feature)

For more detailed analysis, it is possible tor review individual players and their attributes. Data includes many individual attributes over multiple seasons making it possible to follow player's progress or see how switching teams or positions might have affected the player. Data includes many attributes covering accuracy (passing, shooting, free kick), skills (dribbling, ball control), performance (balance, shot power, speed).

#### Data Source

Data set used in this project can be found on Kaggle. It includes data on soccer matches for 11 European leagues and over 25,000 matches covering 2008 to 2016 seasons. 

https://www.kaggle.com/hugomathien/soccer

Additional descriptive league information was scrapped from public online sources.

#### Implementation

The application is developed in **Python**. At this point in development the application does not require saving any data. The data set is stored within a **SQLite database** and is distributed with the application. **SQLAlchemy** toolkit is used to query the dataset and **Pandas** library is used to manipulate the data within the application. **Flask** is used as a framework for user interface. The application is packaged for distribution using **Docker**. 

Prediction model was trained and tested outside of the application code. The model and necessary base features were saved for future use using Python's **Pickle** module. **Scikit-learn** toolkit was used for machine learning element.

#### Match Prediction Model

Within the current data set 46% of matches end up with _Home Win_. So a base model - predicting a _Home Win_ for all matches will have a 46% accuracy. Our research indicates that accuracy can be increased up to 53-54%. There are some project that claim higher accuracy, but they betting odds data and features only available at match conclusion. 

Feature selection for model training took considerable effort. Since the objective was to predict an outcome of future matches, we have decided to base our model only on features available prior to game time. 

Some features we have tried actually lowered the performance. For instance, using team names considerably reduces accuracy when testing (most likely due to overfitting). We had high hopes for some features, but they had little affect. 

Features in our final list fit into 3 categories: 

- **Player ranking**: Teams with more skillful players are more likely to win. Additionally, separating players by position - goalie, forwards, midfielders and defenders - improves model accuracy. 
- **Past team performance**: A team successful in recent past - whether previous match or previous season - is likely to be more energized for their next match.
- **Team attributes**: Similarly to player attributes, team's attributes - play passing, defence aggression, team width - can be used to improve model accuracy. 

Additionally, all features were split between **home** matches and **away** matches since home field advantage applies to any sport. 

Features used for model training:

- `Team Ranking`: Average ranking of all players on the team (with the exception of goalie).
- `Goalie Ranking`: Overall ranking of goalie.
- `Previous Match Result`: Result of last match - Win/Draw/Loss.
- `Winning Rate Last Season`: Winning percentage during previous season (home and away winning percentage is calculated separately).
- `Play Passing`, `Defence Aggression` and `Team Width`: Team attributes that proved helpful in model testing. We have noticed that there is strong correlation between defence performance and positive result. 

TBD: Model implementation/testing

#### Docker Image

The application is available as Docker image from TBD.

#### References

The following two projects based on the same data were reviewed as part of our development. No code was used in our project.

- https://github.com/sgoyal1012/UdacityNanodegreeSubmissions/tree/master/EuropeanSoccer_Capstone
- https://www.kaggle.com/airback/match-outcome-prediction-in-football/

#### Next Steps

TBD