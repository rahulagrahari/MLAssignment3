from configuration import configure
from DatasetTraining import training
from matplotlib import pyplot as plt

# importing the dataset and pre-procesing it

datasetName = 'winequality-white.csv'


# configure constructor takes
#
# datasetName as parameter

cnfg = configure(datasetName)

# binaryClassConversion takes featureName to convert the feature value in two classes: 0 and 1
#
# returns the whole dataset with updated binary values of 0 and 1

dataset = cnfg.binaryClassConversion('quality', 3)

#dataset = cnfg.getdataset()

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = dataset['taste']

classifierList = ["knn", "decisiontree", "logistic"]

training = training(dataset, X, y)

# clf = training.normal_training(classifierList[2], True, 0.3)
# print clf['x_f']
# eval = evalMetric(clf['y_test'], clf['y_pred'])
# print "accuracy_Score for ", classifierList[2], ": ", eval.accuracy_skore()
# print clf['y_pred']

# for cl in classifierList:
#     clf = training.normal_training(cl, True, 0.3)
#     eval = evalMetric(clf['y_test'], clf['y_pred'])
#     print "accuracy_Score for ", cl, ": ", eval.accuracy_skore()
#     print "confusion_metric for ", cl, ": \n", eval.confusion_metric()
#     print "precision_recall_fscore_supports for ", cl, ": ", eval.precision_recall_fscore_supports()[2]
#     # print "area_under_the_curve ", cl, ": ", eval.area_under_the_curve()

# --------------------K-Fold training -------------------
scores = []
for cl in classifierList:
    score, imp, scoreList = training.k_fold_cross_validation(10, cl, 6)
    scores.append(scoreList)

# -------------------------------------rating distribution histograph ----------------------------------------
plt.hist(dataset['quality'], range=(1, 10))
plt.xlabel('Ratings of wines')
plt.ylabel('Amount')
plt.title('Distribution of wine ratings')
plt.show()

# -----------------------performance evaluator----------------------------------------------------------------
# clff = training.normal_training("randomforest", True, 0.3)
# eval = evalMetric(clff['y_test'], clff['y_pred'])
# eval.precision_recall_fscore_supports()
# print clff['trainedmodel'].feature_importances_
# clf = training.k_fold_cross_validation(10, 'randomforest')
# score , fimp = clf
# print sum(fimp)
#
# fig, ax = plt.subplots()
# ax.scatter(clff['y_test'], clff['y_pred'], edgecolors=(0, 0, 0))
# ax.plot([clff['y_test'].min(), clff['y_test'].max()], [clff['y_test'].min(), clff['y_test'].max()], 'k--', lw=4)
# plt.plot(clff['y_test'], clff['y_pred'], color='blue', linewidth=1)
# # plt.plot(clff['X_test'], clff['y_test'], color='yellow', linewidth=0.25)
# plt.show()

# ---------------------------Graph generation--------------------------------------------------------

# ---------------------------feature Importance------------------------------------------------------
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt



score, imp, s1 = training.k_fold_cross_validation(10, 'randomforest')
n_groups = len(imp)
means_men = (imp)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.4
error_config = {'ecolor': '0.5'}
rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Features')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(index + bar_width/2)
plt.legend()
ax.set_xticklabels(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],rotation=60)
plt.tight_layout()
plt.show()

# ----------------------------------------------KNN----------------------------------------------------------
a = []
b = []
import numpy as np
for i in range(1,6):
    score, imp, s1 = training.k_fold_cross_validation(10, 'knn', i*3)
    a.append(i*3)
    b.append(score)
plt.plot(a, b)
plt.axis([2, 16, 0.35, 0.42])
plt.xlabel('neighbours')
plt.ylabel('score')
plt.title('K nearest neighbour')
plt.show()
#--------------------------------------decision tree--------------------------------------

a = []
b = []
for i in range(1,6):
    score, imp, s1 = training.k_fold_cross_validation(10, 'decisiontree', i*3)
    a.append(i*3)
    b.append(score)
plt.plot(a, b)
# plt.axis([2, 16, 0.68, 0.75])
plt.xlabel('depth')
plt.ylabel('score')
plt.title('decision Tree')
plt.show()

# ---------------------------k-fold training map-------------------------------------

plt.plot([i for i in range(1, 11)], scores[0][0], label='knn')
plt.plot([i for i in range(1, 11)], scores[1][0], label='decision tree')
plt.plot([i for i in range(1, 11)], scores[2][0], label='logistic')
plt.xlabel('splits')
plt.ylabel('score')
plt.title('K-fold training map')
plt.legend()
plt.show()
#------------------------accuracy score vs f1_score-------------------------

plt.plot([i for i in range(1, 11)], scores[0][0], label='knn_f1_Score')
plt.plot([i for i in range(1, 11)], scores[0][1], label='knn_accuracy_score')
plt.plot([i for i in range(1, 11)], scores[1][0], label='decision_tree_f1_score')
plt.plot([i for i in range(1, 11)], scores[1][1], label='decision_accuracy_score')
plt.plot([i for i in range(1, 11)], scores[2][0], label='logistic_f1_score')
plt.plot([i for i in range(1, 11)], scores[2][1], label='logistic_accuracy_score')
plt.xlabel('splits')
plt.ylabel('score')
plt.title('accuracy_score vs f1_score for 10 fold splits')
plt.legend(loc=2, prop={'size': 10})
#plt.figure(figsize=(15,15))
plt.show()




