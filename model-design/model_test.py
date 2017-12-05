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
from sklearn.metrics import confusion_matrix

# --------------------------------------------------------------------
# IMPORT DATA
# --------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/akulapa/Data602-Final/master/model-design/matchDf.csv'
matchDf = pd.read_csv(url)

# --------------------------------------------------------------------
# SELECT FIELD FOR TRAINING
# --------------------------------------------------------------------
modelDf = matchDf[[
#                   'home_team_api_id', 'away_team_api_id', 
#                   'stage', 
                   'home_ranking', 'away_ranking',
                   'home_goalie_ranking', 'away_goalie_ranking', 
                   'home_prev', 'away_prev', 
                   'home_win_rate', 'away_win_rate',
#                   'home_play_speed', 'away_play_speed', 
                   'home_play_passing',  'away_play_passing',  
#                   'home_creation_passing', 'home_creation_crossing', 'home_creation_shooting', 
#                   'away_creation_passing', 'away_creation_crossing', 'away_creation_shooting', 
                   'home_aggression', 'away_aggression', 
                   'home_team_width', 'away_team_width', 
                   'result']]
modelDf = modelDf.dropna()

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

print("Accuracy: ", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, clf.predict(X_test)))

# --------------------------------------------------------------------
# SAVE FOR FUTURE USE
# --------------------------------------------------------------------
import pickle

with open('model.pickle', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Test
with open('model.pickle', 'rb') as handle:
    finalCLF = pickle.load(handle)


