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
        dataset = pd.read_csv(self.datasetName, ';')
        return dataset

    def binaryClassConversion(self, featureName):

        dataset = self.getdataset()

        dataset['taste'] = dataset[featureName].apply(isTasty)
        return dataset
