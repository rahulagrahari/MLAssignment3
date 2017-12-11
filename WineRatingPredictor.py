import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler



# importing the dataset and pre-procesing it
datasetName = raw_input("which data set: please type red or white")
dataset = pd.read_csv('winequality-'+datasetName+'.csv', ';')


def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


dataset['taste'] = dataset['quality'].apply(isTasty)
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = dataset['taste']

# splitting into training and test dataset


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# [subset.shape for subset in [X_train, X_test, y_train, y_test]]

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# -------Training the data set------------------------------------------------------------------------------------------

# decision tree
simpleTree = DecisionTreeClassifier(max_depth=5)
simpleTree.fit(X_train, y_train)
y_pred_dTree = simpleTree.predict(X_test)
accuracy_score(y_test, y_pred_dTree)

# GradientBoostingClassifier
gbmTree = GradientBoostingClassifier(max_depth=5)
gbmTree.fit(X_train, y_train)
y_pred_gbmTree = gbmTree.predict(X_test)
accuracy_score(y_test, y_pred_gbmTree)

# Random Forest
rfTree = RandomForestClassifier(max_depth=15)
rfTree.fit(X_train, y_train)
y_pred_rfree = rfTree.predict(X_test)
accuracy_score(y_test, y_pred_rfree)

# support vector
supportVector = SVC(kernel='poly', random_state=0, probability=True)
supportVector.fit(X_train, y_train)
y_pred_sVector = supportVector.predict(X_test)
supportVector.score(X_test, y_test)
accuracy_score(y_test, y_pred_sVector)
# -----------------------------------------------------------------------------------------------------------------

# performance evaluator

simpleTreePerformance = precision_recall_fscore_support(y_test, simpleTree.predict(X_test))

rfTreePerformance = precision_recall_fscore_support(y_test, rfTree.predict(X_test))

gbmTreePerformance = precision_recall_fscore_support(y_test, gbmTree.predict(X_test))

supportVectorPerformance = precision_recall_fscore_support(y_test, supportVector.predict(X_test))


# ----------------------------------------------------------------------------------------------------------------------


