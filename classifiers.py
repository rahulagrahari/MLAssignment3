class classifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def decision_tree(self):
        from sklearn.tree import DecisionTreeClassifier
        simpleTree = DecisionTreeClassifier(max_depth=5)
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
        supportVector = SVC(kernel='poly', random_state=0, probability=True)
        return supportVector.fit(self.X, self.y)