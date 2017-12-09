### Data 602 Final Project: SOCCER DATA ANALYSIS TOOL
### CUNY MSDS Program, Fall 2017
### Pavan Akula, Nnaemezue Obi-eyisi, Ilya Kats


#### Objective

The goal of this proof-of-concept project is to design an application that allows analysis of soccer data. The application should include descriptive and statistical features as well as predictive model.

#### Summary

The app proved a successful concept. Since it is designed around idea of analyzing past match data and predicting outcome of future matches, it can be potentially used in several areas - assisting with assignment of odds in sports betting, assisting with designing fantasy league teams, assisting fans to better understand and connect with the game. 

We picked up several interesting findings working on this project:

- There is interesting correlation between some team attributes and match outcome. Unsurprisingly, _Shots On Goal_ are positively correlated with wins. Defensive attributes also have noticeable positive correlation with wins. More surprisingly, attributes with creative element - creative passing, crossing and shooting - often have negative correlation with wins, so the data shows that a fundamental approach is more rewarding. However, there is a lot of variation in these attributes, so their influence needs to be better studied.
- Attributes of individual players differ based on their position. It is sometimes possible to observe a drop or increase in certain attributes as the player switches positions. Skills required for defenders are not the same as skills required for forwards.
- It appears that there are many factors that affect match outcome, but they each have a fairly small influence. There is no one feature that will reliably predict the outcome. This gives a certain element of randomness to a match between two teams that are on a similar level, which arguably makes the sport more exciting.

#### A Note about Soccer Data

Out of major sports, soccer probably has fewest data points that can be measured objectively. It is a long, low-scoring game where a single mistake can make or break it thus increasing randomness of outcomes. Individual and team skill levels have a great affect on game outcome, but they are also very subjective. Our data set uses FIFA (International Federation of Association Football) statistics for team and player attributes that was originally scraped from official FIFA video game. It is unlikely that we will experience the Moneyball moment in soccer in the near future.

#### User Interface/App Functionality

To start a user picks a league, then a season, and then a team. The user will be presented with the information relevant to the team's performance during selected season. The following information is presented: 

- Player list, including age, height and weight
- List of matches with outcomes
- Goal and foul counts per match
- Shooting comparison per match that includes _On Goal_ shots and _Off Goal_ shots
- Top players in the following categories: Goals, Shots, Fouls and Cards
- Correlation between team attributes and match outcome (attributes cover various categories such as team's play, shooting, defense, and attack)
- Prediction for match outcomes against other teams in the league (see _Model_ section below for more information about predictive feature)

For more detailed analysis, it is possible to review individual players and their attributes. Data includes many individual attributes over multiple seasons making it possible to follow player's progress or see how switching teams or positions might have affected the player. Data includes many attributes covering accuracy (passing, shooting, free kick), skills (dribbling, ball control), performance (balance, shot power, speed).

#### Data Source

Data set used in this project can be found on Kaggle. It includes data on soccer matches for 11 European leagues and over 25,000 matches covering 2008 to 2016 seasons. 

https://www.kaggle.com/hugomathien/soccer

Additional descriptive league information was scrapped from public online sources.

#### Implementation

The application is developed in **Python**. At this point in development the application does not require saving any data. The data set is stored within a **SQLite database** and is distributed with the application. **SQLAlchemy** toolkit is used to query the dataset and **Pandas** library is used to manipulate the data within the application. **Flask** is used as a framework for user interface. The application is packaged for distribution using **Docker**. 

Prediction model was trained and tested outside of the application code. The model and necessary base features were saved for future use using Python's **Pickle** module. **Scikit-learn** toolkit was used for machine learning element.

Key files and folders:

- `app\dbfunctions.py` file contains code to manipulate app data.
- `app\model_test.py` file was used to test performance of various classifiers.
- `app\model_training.py` file was used to design and test prediction features.
- `app\models.py` file contains classes used to store app data.
- `app\prediction.py` file contains code to run match outcome prediction.
- `app\views.py` file uses Flask to render HTML pages.
- `database` folder contains SQLite database. Please note that the database **must be unzipped** before running the app.
- `instance` folder contains prediction model and base features saved using `pickle` module.
- `templates` folder contains HTML templates.

#### Match Prediction Model

Within the current data set 46% of matches end up with _Home Win_. So a base model - predicting a _Home Win_ for all matches will have a 46% accuracy. Our research indicates that accuracy can be increased up to 53-54%. There are some projects that claim higher accuracy, but they use betting odds data and features only available at match conclusion. 

Feature selection for model training took considerable effort. Since the objective was to predict an outcome of future matches, we have decided to base our model only on **features available prior to match**. 

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

The model needs to predict 3 classes - Home Win, Draw and Home Loss. We have tried several classifiers - SVM, kNN, Gaussian Naive Bayes and Adaptive Boosting. **AdaBoost** with default settings (`sciki-learn` toolkit) provided the best results.

Only matches with all features available were used in machine learning. This set includes 13,128 matches. 20% of matches data were set aside for model testing. Depending on the random sample selected for training, we saw accuracy results as high as 52.7%, but it averaged about 51%. 

Confusion matrix based on test data:

| Home Team Result | Predicted: Win | Predicted: Draw | Predicted: Loss | Total |
|------------------|----------------|-----------------|-----------------|-------|
| Actual: Win      | 993            | 2               | 236             | 1,231 |
| Actual: Draw     | 467            | 4               | 178             | 649   |
| Actual: Loss     | 398            | 3               | 345             | 746   |
| Total            | 1,858          | 9               | 759             | 2,626 |

_Home Win_ category is the most reliable - accurately predicted 80.67% of the time. _Home Loss_ is accurately predicted only 46.25% of the time. _Draw_ is very difficult to predict (we saw similar findings in other projects). The model rarely picks it. Probabilities assigned to classes during prediction often differ by less than 1% - they hover around 32-34% per class.

Since this is a proof-of-concept project, we went with a simple implemtation of the model within the app. When running a prediction between two teams in a season we are using features of the last match for each team in the season. We believe this is representative of team's performance during the time frame we are interested in. Some features, such as list of players, are displayed along with model's prediction. In future development of the app, it will be possible to modify features to compare predictions or possibly run multiple predictions simultaneously across a range of values.

#### Docker Image

The application is available as Docker image from https://hub.docker.com/r/mezulity/football_flask/.

#### References

The following two projects based on the same data were reviewed as part of our development. No code was used in our project.

- https://github.com/sgoyal1012/UdacityNanodegreeSubmissions/tree/master/EuropeanSoccer_Capstone
- https://www.kaggle.com/airback/match-outcome-prediction-in-football/
