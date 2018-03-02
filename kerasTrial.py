# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

def MLTestTrainSplit(epochs):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # load pima indians dataset
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=seed)
    # create model
    layers = [Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'),\
              Dense(8, kernel_initializer='uniform', activation='relu'),\
              Dense(1, kernel_initializer='uniform', activation='sigmoid')]
    model = Sequential(layers)
    #kernel_initializer sets initial weights (uniform or normal)
    #activation sets type of activation function (tanh, sigmoid, linear, elu, selu, softplus, softsign)
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    hist = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=epochs, batch_size=10,  verbose=0)
    print(hist.history)
    # calculate predictions
    predictions = model.predict(X_test)
    # round predictions
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
    predictions = [round(x[0]) for x in predictions]
    return y_test, predictions

def confusionMatrixPlot(y_test, prediction, fig_num=2):
    """
    Plots proportion of right and wrong predictions in a heat map
    """
    cm = metrics.confusion_matrix(y_test, prediction)
    #print correct percentage
    print("Percentage of correct predictions is %1.2f" % ((cm[0][0] + cm[1][1])/(sum(sum(cm)))*100.))
    plt.figure(fig_num, figsize=(12,12))
    sns.heatmap(cm, annot=True, fmt="d", )
    plt.ylabel("Real value")
    plt.xlabel("Predicted value")

def plotEpochGraph():    
    Tlist = []
    elist = []
    e = np.arange(50, 300, 20)
    print(e)
    for i in range(len(e)):
        print(e[i])
        T, F = MLTestTrainSplit(e[i])
        elist.append(e[i])
        Tlist.append(T/(T+F))
    
    plt.plot(elist, Tlist)
    plt.show()

def MLKFoldCrossValid(epoch):
    seed = 7
    np.random.seed(seed)
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    
    for train, test in kfold.split(X,Y):
        layers = [Dense(12, input_dim=8, activation='relu', kernel_initializer='uniform'),\
                  Dense(8,activation='relu', kernel_initializer='uniform'),\
                  Dense(1,activation='sigmoid', kernel_initializer='uniform')]
        model = Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X[train], Y[train], epochs = epoch, batch_size=10, verbose=0)
        scores = model.evaluate(X[test], Y[test], verbose = 0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1]*100)
    print("%.2f%%(+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

y_test, prediction = MLTestTrainSplit(1000)
confusionMatrixPlot(y_test, prediction, fig_num=1)
plt.show()