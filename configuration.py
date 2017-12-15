def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


def levelOfTaste(quality):
    if quality >= 7:
        return 1
    elif quality < 7 and quality >=5:
        return 0
    else:
        return -1


class configure:

    def __init__(self, dataseetName):
        self.datasetName = dataseetName

    def getdataset(self):
        import pandas as pd
        dataset = pd.read_csv(self.datasetName, ';')
        return dataset

    def binaryClassConversion(self, featureName, split=2):

        dataset = self.getdataset()

        if split == 2:
            dataset['taste'] = dataset[featureName].apply(isTasty)
        else:
            dataset['taste'] = dataset[featureName].apply(levelOfTaste)
        return dataset
