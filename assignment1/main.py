import pandas as pd
from decisionTree import decisionTree
from regression import regression

def loadData(file):
    data = pd.read_csv(file, header=None)
    return data.drop_duplicates()

def transformMushroomTargetAttribute(dataSet, targetAttribute):
    uniqueValues = dataSet[targetAttribute].unique()
    dataSet[targetAttribute].replace(
        uniqueValues, range(len(uniqueValues)), inplace=True)
    return dataSet

if __name__ == '__main__':

    print("1.1.a")
    decisionTree(loadData('./data/iris.csv'), [0.05, 0.10, 0.15, 0.20], True).validate()

    print("1.1.b")
    decisionTree(loadData('./data/spambase.csv'), [0.05, 0.10, 0.15, 0.20, 0.25], True).validate()
    
    print("2.1.a")
    data = loadData('./data/mushroom.csv')
    target = len(data.columns) - 1
    vals = data[target].unique()
    data[target].replace(vals, range(len(vals)), inplace=True)
    decisionTree(data, [0.05, 0.10, 0.15], False).validate()

    print("2.1.b")
    newData = pd.get_dummies(data=data, columns=range(target))
    newTarget = newData[target]
    newData.drop(labels=[target], axis=1, inplace=True)
    newData.insert(len(newData.columns), target, newTarget)
    decisionTree(newData, [0.05, 0.10, 0.15], True).validate()

    print("6.a")
    regression(loadData('./data/housing.csv'), [0.05, 0.10, 0.15, 0.20]).validate()
