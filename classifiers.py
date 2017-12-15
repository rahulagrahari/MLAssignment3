from sklearn.feature_selection import RFE
class classifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y



    def decision_tree(self, n):
        from sklearn.tree import DecisionTreeClassifier
        simpleTree = DecisionTreeClassifier(max_depth=9)
        return simpleTree.fit(self.X, self.y)

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        simpleTree = RandomForestClassifier(max_depth=5)
        return simpleTree.fit(self.X, self.y)

    def gradient_boosting(self):
        from sklearn.ensemble import GradientBoostingClassifier
        simpleTree = GradientBoostingClassifier(max_depth=5)
        return simpleTree.fit(self.X, self.y)

    def support_vector(self):
        from sklearn.svm import SVC
        supportVector = SVC(kernel='linear', random_state=0, probability=True)
        return supportVector.fit(self.X, self.y)

    def linear_support_vector(self):
        from sklearn.svm import LinearSVC
        supportVector = LinearSVC(random_state=0)
        return supportVector.fit(self.X, self.y)

    def logistic(self):
        from sklearn.linear_model import LogisticRegression
        linear = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        return linear.fit(self.X, self.y)

    def KNN(self, n):
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(3)
        return knn.fit(self.X, self.y)

    def sgd(self):
        from sklearn.linear_model import SGDRegressor
        sgd = SGDRegressor()
        return sgd.fit(self.X, self.y)