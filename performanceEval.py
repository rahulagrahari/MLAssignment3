class evalMetric:

    def __init__(self, n_split=2, dataset=None, classifier=None, X=None, y=None):
        self.n_split = n_split
        self.dataset = dataset
        self.classifier = classifier
        self.X = X
        self.y = y

    def k_fold_cross_validation(self):

        from sklearn.model_selection import KFold
        import numpy as np
        from classifiers import classifier
        from sklearn.metrics import accuracy_score
        kf = KFold(n_splits=self.n_split)
        X = np.array(self.X)
        y = np.array(self.y)
        dataset = np.array(self.dataset)
        score = 0
        for train_indices, test_indices in kf.split(dataset):
            x_train = [X[i] for i in train_indices]
            x_test = [X[i] for i in test_indices]
            y_train = [y[i] for i in train_indices]
            y_test = [y[i] for i in test_indices]
            clf = classifier(x_train, y_train)
            if self.classifier == "randomforest":
                predict = clf.random_forest().predict(x_test)
            elif self.classifier == "svm":
                predict = clf.support_vector().predict(x_test)
            elif self.classifier == "gradientboosting":
                predict = clf.gradient_boosting().predict(x_test)
            elif self.classifier == "decisiontree":
                predict = clf.decision_tree().predict(x_test)
            score += accuracy_score(y_test, predict)
            print("Score: ", score)
        print("Average Score: ", score / self.n_split)
        return score / self.n_split


