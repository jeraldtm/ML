'''
Created on 1 Mar 2018

@author: jerald
ML processing of binary classification type Adult data set using Logistic Regression
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import ceil
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

def plotData(data):
    """
    Plots all features from dataset
    """
    fig = plt.figure(figsize=(20,15))
    cols = 5
    rows = ceil(float(data.shape[1]) / cols)
    for i, column in enumerate(data.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if data.dtypes[column] == np.object:
            data[column].value_counts().plot(kind="bar", axes=ax)
        else:
            data[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()

# 
def processing(inputdata, processingType='number'):
    """
    Encode the categorical features
    processingType  : 'number' converts features to a number
                    : 'binary' converts each individual possibility into a feature, with value 0 or 1
    """
    inputdata2 = inputdata.copy()
    encoders = {}
    for column in inputdata.columns:
        if inputdata.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            inputdata[column] = encoders[column].fit_transform(inputdata[column].fillna('0'))
    
    if (processingType=='number'):      
        number_target = inputdata["Target"]
        del inputdata["Target"]
        return inputdata, number_target, encoders
            
    elif (processingType=='binary'):
        binary_data = pd.get_dummies(inputdata2)
        # Let's fix the Target as it will be converted to dummy vars too
        binary_target = binary_data["Target_>50K"]
        del binary_data["Target_<=50K"]
        del binary_data["Target_>50K"]
        return binary_data, binary_target, encoders
    
def splitAndScale(features, target, scaled=0):
    """
    Split data set into train and test
    scaled  :0:no scaling on X_train and X_test
            :1: scaling on X_train and X_test using X_train mean and std
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.80)
    #Scale using X_train statistics
    if (scaled==1):
        scaler = preprocessing.StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float64')),columns=X_train.columns)
        X_test = scaler.transform(X_test.astype('float64'))
    return X_train, X_test, y_train, y_test

def fitModelPrediction(train, test):
    """
    Creates Logistic Regression model
    Trains on X_train and y_train values, and
    gives prediction of y_test values for X_test dataset.
    """
    cls = linear_model.LogisticRegression()
    cls.fit(train, y_train)
    prediction = cls.predict(test)
    return cls, prediction

##########Plotting functions###########
def confusionMatrixPlot(y_test, prediction, fig_num=2):
    """
    Plots proportion of right and wrong predictions in a heat map
    """
    cm = metrics.confusion_matrix(y_test, prediction)
    #print correct percentage
    print("Percentage of correct predictions is %1.2f" % ((cm[0][0] + cm[1][1])/(sum(sum(cm)))*100.))
    plt.figure(fig_num, figsize=(12,12))
    sns.heatmap(cm, annot=True, xticklabels=encoders["Target"].classes_, yticklabels=encoders["Target"].classes_, fmt="d", )
    plt.ylabel("Real value")
    plt.xlabel("Predicted value")

def f1Plot(cls, prediction, train, y_test, fig_num=3):
    """
    Plots the contribution of each feature to the prediction
    """
    metrics.f1_score(y_test, prediction)
    coefs = pd.Series(cls.coef_[0], index=train.columns).sort_values(ascending = True)
    print(coefs)
    plt.figure(fig_num)
    coefs.plot(kind="bar")

#####################################Main######################################
#Input data, remove fnlwgt and Education data#    
data = pd.read_csv("adult.csv", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Maritial Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"], sep=r'\s*,\s*', engine='python', na_values="?")
data = data.drop("fnlwgt",1)
data = data.drop("Education",1)

#Encode data, convert labels to numbers, extract features from data
features, target, encoders = processing(data, processingType='binary')

#Plot heatmap
sns.heatmap(features.corr(), square=True, cmap="YlGnBu", center = 0)

#Split dataset
X_train_scaled, X_test_scaled, y_train, y_test = splitAndScale(features, target, 1)    

#Set up and train model using scaled X_train values
cls, prediction = fitModelPrediction(X_train_scaled, X_test_scaled)

#Plot prediction Results
confusionMatrixPlot(y_test, prediction, 2)
f1Plot(cls, prediction, X_train_scaled, y_test, 3)
plt.show('all')
