from classifiers import classifier


class training:

    def __init__(self, dataset=None, X=None, y=None):

        self.dataset = dataset
        self.X = X
        self.y = y

    def k_fold_cross_validation(self, n_split=2, classifierName=None):

        from sklearn.model_selection import KFold
        import numpy as np
        from evalMetrics import evalMetric

        kf = KFold(n_splits=n_split)
        X = np.array(self.X)
        y = np.array(self.y)
        dataset = np.array(self.dataset)
        score = 0
        feature_imp = 0
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
                c = clf.decision_tree()
                predict = c.predict(x_test)
            elif classifierName == "linearsvm":
                c = clf.linear_support_vector()
                predict = c.predict(x_test)

            score += evalMetric(y_test, predict).precision_recall_fscore_supports()[2]
            feature_imp += c.feature_importances_
            # print("Score: ", score)

        print("Average Score: ", score / n_split)
        return score / n_split, feature_imp / n_split

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
            trainedClassifier = clf.random_forest()
            y_pred = trainedClassifier.predict(X_test)
        elif classifierName == "svm":
            trainedClassifier = clf.support_vector()
            y_pred = trainedClassifier.predict(X_test)
        elif classifierName == "gradientboosting":
            trainedClassifier = clf.gradient_boosting()
            y_pred = trainedClassifier.predict(X_test)
        elif classifierName == "decisiontree":
            trainedClassifier = clf.decision_tree()
            y_pred = trainedClassifier.predict(X_test)
        elif classifierName == "linearsvm":
            trainedClassifier = clf.linear_support_vector()
            y_pred = trainedClassifier.predict(X_test)
        dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "trainedmodel": trainedClassifier,
                "y_pred": y_pred}

        return dict


