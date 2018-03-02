'''
Created on 1 Mar 2018

@author: jerald

Regression type dataset wine quality ML using Random Forest Regressor
'''
import sklearn
import numpy as np 
import pandas as pd #Data processing array handling module
from sklearn.model_selection import train_test_split #Library to help select model
from sklearn import preprocessing #Preprocesses our data
from sklearn.ensemble import RandomForestRegressor #Import family of models
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import GridSearchCV #Modules for cross validation
from sklearn.metrics import mean_squared_error, r2_score #Metrics to judge performance of model
from sklearn.externals import joblib #Module to save our model

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

def printData():
    print(data.head())
    print(data.shape)
    print(data.describe())
    
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y) #20% of data for testing, set random state seed as 123, stratify by target variable to ensure similar distribution between test and training set 

def scaleData(X_train):
    """ To scale data to be centred around zero, not suitable in our case for features with different scales
    We won't be able to perform exact same transformation on the test set. Can scale separately, but since test set
    has different means and std, it won't be a fair representation of how the pipeline would perform on new data. 
    """
    X_train_scaled = preprocessing.scale(X_train)
    print(X_train_scaled)

def TransformData():
    """Instead use Transformer API, where a preprocessing step can be fit to the data using the training data. The same
        transformation would then be used on future data sets.
        1) Fit transformer on training set, saving the mean and std.
        2) Apply transformer on training set, scaling the training data.
        3) Apply transformer to test set using same means and std. 
    """
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
#Declare pipeline
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100)) 
   
#Declare hyperparameters, do Cross-validation for different hyperparameters
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5,3,1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(r2_score(y_test, prediction))

#Save model
joblib.dump(clf, 'rf_regressor.pkl')

#Load model
clf2 = joblib.load('rf_regressor.pkl')
prediction2 = clf2.predict(X_test)

