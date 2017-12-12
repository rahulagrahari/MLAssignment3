from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from configuration import configure
from classifiers import classifier
from performanceEval import evalMetric

# importing the dataset and pre-procesing it
datasetName = input("which data set: please type red or white")
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

# -----------------------performance evaluator----------------------------------------------------------------
classifierList = ["svm", "gradientboosting", "randomforest", "decisiontree" ]
eval1 = evalMetric(10, dataset, classifierList[0], X, y)
eval1.k_fold_cross_validation()
eval2 = evalMetric(10, dataset, classifierList[1], X, y)
eval2.k_fold_cross_validation()
eval3 = evalMetric(10, dataset, classifierList[2], X, y)
eval3.k_fold_cross_validation()
eval4 = evalMetric(10, dataset, classifierList[3], X, y)
eval4.k_fold_cross_validation()




