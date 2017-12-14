from configuration import configure
from DatasetTraining import training
from matplotlib import pyplot as plt
from evalMetrics import evalMetric


# importing the dataset and pre-procesing it

datasetName = 'winequality-white.csv'


# configure constructor takes
#
# datasetName as parameter

cnfg = configure(datasetName)

# binaryClassConversion takes featureName to convert the feature value in two classes: 0 and 1
#
# returns the whole dataset with updated binary values of 0 and 1

# dataset = cnfg.binaryClassConversion('quality')

dataset = cnfg.getdataset()

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = dataset['quality']


classifierList = [ "gradientboosting", "randomforest", "decisiontree", "linearsvm"]

training = training(dataset, X, y)

for cl in classifierList:
    clf = training.normal_training(cl, True, 0.3)
    eval = evalMetric(clf['y_test'], clf['y_pred'])
    print "accuracy_Score for ", cl, ": ", eval.accuracy_skore()
    print "confusion_metric for ", cl, ": \n", eval.confusion_metric()
    print "precision_recall_fscore_supports for ", cl, ": ", eval.precision_recall_fscore_supports()[2]
    # print "area_under_the_curve ", cl, ": ", eval.area_under_the_curve()

for cl in classifierList:
    clf = training.k_fold_cross_validation(10, cl)

# -------------------------------------rating distribution graph ----------------------------------------
plt.hist(dataset['quality'], range=(1, 10))
plt.xlabel('Ratings of wines')
plt.ylabel('Amount')
plt.title('Distribution of wine ratings')
plt.show()

# -----------------------performance evaluator----------------------------------------------------------------
clff = training.normal_training("randomforest", True, 0.3)
eval = evalMetric(clff['y_test'], clff['y_pred'])
eval.precision_recall_fscore_supports()
print clff['trainedmodel'].feature_importances_
clf = training.k_fold_cross_validation(10, 'randomforest')
score , fimp = clf
print sum(fimp)

fig, ax = plt.subplots()
ax.scatter(clff['y_test'], clff['y_pred'], edgecolors=(0, 0, 0))
ax.plot([clff['y_test'].min(), clff['y_test'].max()], [clff['y_test'].min(), clff['y_test'].max()], 'k--', lw=4)
plt.plot(clff['y_test'], clff['y_pred'], color='blue', linewidth=1)
# plt.plot(clff['X_test'], clff['y_test'], color='yellow', linewidth=0.25)
plt.show()


