import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

os.listdir(os.getcwd())

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# categorize info
factors = [
    'season',
    'holiday',
    'workingday',
    'weather'
]

for factor in factors:
    train[factor] = train[factor].astype('category')
    test[factor] = test[factor].astype('category')

# process datetime
train.datetime = pd.to_datetime(train.datetime, format='%Y-%m-%d %H:%M:%S')
test.datetime = pd.to_datetime(test.datetime, format='%Y-%m-%d %H:%M:%S')

# additional features
train['hour'] = train.datetime.map(lambda x: x.hour)  # map lambda is basically sapply; hour method only works on single entry
train['weekday'] = train.datetime.map(lambda x: x.dayofweek)
train['year'] = train.datetime.map(lambda x: x.year)

test['hour'] = test.datetime.map(lambda x: x.hour)
test['weekday'] = test.datetime.map(lambda x: x.dayofweek)
test['year'] = test.datetime.map(lambda x: x.year)

add_features = ['hour', 'weekday', 'year']
for feature in add_features:
    train[feature] = train[feature].astype('category')
    test[feature] = test[feature].astype('category')

# define training and testing sets
x_train = train.drop('casual', axis=1).drop('registered', axis=1).drop('count', axis=1).copy()
y_train_casual = train['casual']
x_test = test.copy()

# random forest
rfc = RandomForestClassifier(n_estimators=40, max_features=4)
rfc.fit(x_train, y_train_casual)

sns.distplot(train.weekday)
