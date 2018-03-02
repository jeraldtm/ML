'''
Created on 1 Mar 2018

@author: jerald

Classification type dataset ML using Random Forest Classifier
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# sklearn provides the iris species as integer values since this is required for classification
# here we're just adding a column with the species names to the dataframe for visualisation
df['species'] = np.array([iris.target_names[i] for i in iris.target])
# sns.pairplot(df, hue='species')
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.5, stratify=iris.target, random_state=123456)
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456) #Uses RandomForestClassifer with 100 trees
rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')

cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
sns.heatmap(cm, annot=True)
plt.show() 