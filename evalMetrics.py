from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score, f1_score
class evalMetric:

    def __init__(self, y_test=None, y_pred=None):

        self.y_test = y_test
        self.y_pred = y_pred

    def precision_recall_fscore_supports(self):
        return precision_recall_fscore_support(self.y_test, self.y_pred, average='weighted')

    def confusion_metric(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def area_under_the_curve(self):
        return roc_auc_score(self.y_test, self.y_pred)

    def accuracy_skore(self):
        return accuracy_score(self.y_test, self.y_pred)

    def F1_score(self):
        return f1_score(self.y_test, self.y_pred)



