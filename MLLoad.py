'''
Created on 28 Feb 2018

@author: jerald
'''
import numpy as np
from keras.models import load_model

def MLLoad(modelName):
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    model = load_model(modelName)
    predictions = model.predict(X)
    rounded = [round(x[0]) for x in predictions]
    T = 0
    F = 0
    for i in range(len(rounded)):
        if rounded[i] == Y[i]:
            T+=1
        else:
            F+=1
    return T,F

T, F = MLLoad('my_model.h5')
print(T/(T+F))