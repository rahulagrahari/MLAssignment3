from classifiers import classifier


class training:

    def __init__(self, dataset=None, X=None, y=None):

        self.dataset = dataset
        self.X = X
        self.y = y

    def k_fold_cross_validation(self, n_split=2, classifierName=None, n=0):

        from sklearn.model_selection import KFold
        import numpy as np
        from evalMetrics import evalMetric

        kf = KFold(n_splits=n_split)
        X = np.array(self.X)
        y = np.array(self.y)
        dataset = np.array(self.dataset)
        score = 0
        feature_imp = 0
        scores = []
        scores1 = []
        score1 = []
        score2 = []
        for train_indices, test_indices in kf.split(dataset):
            x_train = X[train_indices]
            x_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            clf = classifier(x_train, y_train)
            if classifierName == "randomforest":
                c = clf.random_forest()
                predict = c.predict(x_test)
            elif classifierName == "svm":
                c = clf.support_vector()
                predict = c.predict(x_test)
            elif classifierName == "gradientboosting":
                c = clf.gradient_boosting()
                predict = c.predict(x_test)
            elif classifierName == "decisiontree":
                c = clf.decision_tree(n)
                predict = c.predict(x_test)
            elif classifierName == "linearsvm":
                c = clf.linear_support_vector()
                predict = c.predict(x_test)
            elif classifierName == "logistic":
                c = clf.logistic()
                predict = c.predict(x_test)
            elif classifierName == "knn":
                c = clf.KNN(n)
                predict = c.predict(x_test)
            elif classifierName == "sgd":
                c = clf.sgd()
                predict = c.predict(x_test)

            e1 = evalMetric(y_test, predict).F1_score()
            score1.append(e1)
            e = evalMetric(y_test, predict).accuracy_skore()
            score2.append(e)
            # scores1.append([[e], [e1]])
            score += e1

            if classifierName == 'randomforest':
                feature_imp += c.feature_importances_
            # print("Score: ", score)

        print("Average Score: ", score / n_split)
        scores.append(score1)
        scores.append(score2)
        return score / n_split, feature_imp / n_split, scores

    def normal_training(self, classifierName=None, featureScaling = False, test_size=0.2):

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        trainedClassifier = None
        y_pred = None
        if featureScaling:
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
        clf = classifier(X_train, y_train)
        if classifierName == "randomforest":
            trainedClassifier, x_f = clf.random_forest()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "svm":
            trainedClassifier = clf.support_vector()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "gradientboosting":
            trainedClassifier = clf.gradient_boosting()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "decisiontree":
            trainedClassifier, x_f = clf.decision_tree()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "linearsvm":
            trainedClassifier = clf.linear_support_vector()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "logistic":
            trainedClassifier = clf.logistic()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "knn":
            trainedClassifier = clf.KNN()
            y_pred = trainedClassifier.predict(X_test)

        elif classifierName == "sgd":
            trainedClassifier = clf.sgd()
            y_pred = trainedClassifier.predict(X_test)

        else:
            print "classifier not found"
        dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "trainedmodel": trainedClassifier,
                "y_pred": y_pred, "x_f":x_f}

        return dict


