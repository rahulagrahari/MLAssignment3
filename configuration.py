def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


class configure:

    def __init__(self, dataseetName):
        self.datasetName = dataseetName

    def getdataset(self):
        import pandas as pd
        dataset = pd.read_csv('winequality-' + self.datasetName + '.csv', ';')
        return dataset

    def binaryClassConversion(self):

        dataset = self.getdataset()

        dataset['taste'] = dataset['quality'].apply(isTasty)
        return dataset
