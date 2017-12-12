
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from configuration import configure
from classifiers import classifier

# importing the dataset and pre-procesing it
datasetName = raw_input("which data set: please type red or white")
cnfg = configure(datasetName)
dataset = cnfg.binaryClassConversion()
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

clf = classifier(X_train, y_train)

# decision tree
accuracy_score(y_test, clf.decision_tree().predict(X_test))

# GradientBoostingClassifier
accuracy_score(y_test, clf.gradient_boosting().predict(X_test))

# Random Forest
accuracy_score(y_test, clf.random_forest().predict(X_test))

# support vector
accuracy_score(y_test, clf.support_vector().predict(X_test))
# -----------------------------------------------------------------------------------------------------------------

# performance evaluator

# ----------------------------------------------------------------------------------------------------------------------
kf = KFold(n_splits=1)
print kf
score = 0
import numpy as np
dataset = np.array(dataset)
X = np.array(X)
y = np.array(y)
for train_indices, test_indices in kf.split(dataset):
    x_train = [X[i] for i in train_indices]
    x_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    clf = classifier(x_train, y_train)
    score += accuracy_score(y_test, clf.gradient_boosting().predict(x_test))

from performanceEval import evalMetric

eval = evalMetric(2,dataset,"randomforest",X,y)
eval.k_fold_cross_validation()
