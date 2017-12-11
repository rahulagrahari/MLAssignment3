import pandas as pd
class config:

    def __init__(self, dataseetName):
        self.datasetName = dataseetName

    def getdataset(self):
        dataset = pd.read_csv('winequality-' + self.datasetName + '.csv', ';')
        return dataset

    def isTasty(quality):
        if quality >= 7:
            return 1
        else:
            return 0

    def binaryClassConversion(self):
        dataset = self.getdaset
        dataset['taste'] = dataset['quality'].apply(self.isTasty)
