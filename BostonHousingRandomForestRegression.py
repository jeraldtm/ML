'''
Created on 1 Mar 2018

@author: jerald
Calculates fit of Random Forest Regression to Boston Housing Data with and without scaling, and with and without 50k cutoff
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

#Check if possible to reduce dimensionality of problem
def dimensionalReductionAnalysis(X_train, X_train_scaled):
    pca = PCA()
    pca.fit(X_train)
    cpts = pd.DataFrame(pca.transform(X_train))
    x_axis = np.arange(1,pca.n_components_+1)
    pca_scaled = PCA()
    pca_scaled.fit(X_train_scaled)
    cpts_scaled = pd.DataFrame(pca_scaled.transform(X_train_scaled))
    
    pcaNorm = pca.explained_variance_/sum(pca.explained_variance_)
    pca_scaledNorm = pca_scaled.explained_variance_/sum(pca_scaled.explained_variance_)
    
    xlist = range(len(pcaNorm))
    plt.figure(1)
    plt.bar(xlist, pcaNorm)
    plt.figure(2)
    plt.bar(xlist, pca_scaledNorm)
    plt.show()

def remove50k(features, targets):
    list = []
    for x in range(len(targets)):
        if targets[x] == 50.:
            list.append(x)
    
    targets = np.delete(targets, list)
    
    for i in list:
        features.drop(features.index[i], inplace=True)
        
    return features, targets

def ML(features, targets, fig_num):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=42)
    
    #Preprocessing
    scaler = StandardScaler().fit(X_train)
    #Scale and construct new dataframes from sklearn numpy array output
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index.values, columns=X_train.columns.values)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index.values, columns=X_test.columns.values)
    
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rf.fit(X_train, y_train)
    predicted_test = rf.predict(X_test)
    
    rf.fit(X_train_scaled, y_train)
    predicted_test_scaled = rf.predict(X_test_scaled)
    
    test_score = r2_score(y_test, predicted_test)
    test_score_scaled = r2_score(y_test, predicted_test_scaled)
    spearman = spearmanr(y_test, predicted_test)
    spearman_scaled = spearmanr(y_test, predicted_test_scaled)
    pearson = pearsonr(y_test, predicted_test)
    pearson_scaled = pearsonr(y_test, predicted_test_scaled)    
    print("R-squared: %1.4f, Scaled R-squared: %1.4f, \n Spearman: %1.4f, Scaled Spearman: %1.4f \n Pearson: %1.4f, Scaled Pearson: %1.4f" % (test_score, test_score_scaled, spearman[0], spearman_scaled[0],pearson[0], pearson_scaled[0]))
    
    plt.figure(fig_num)
    plt.scatter(y_test, predicted_test, label='unscaled')
    plt.scatter(y_test, predicted_test_scaled, label='scaled')
    plt.legend()    

#Main#
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
print('With 50k')
ML(features, targets, 1)

features_wo50k, targets_wo50k = remove50k(features, targets)
print('Without 50k')
ML(features_wo50k, targets_wo50k, 2)

plt.show()