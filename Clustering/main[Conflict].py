
import pandas as pd
from scipy.stats import zscore
from KMeans import KMeans

def loadData(file):
    data = pd.read_csv(file, header=None)
    data = data.fillna(0).drop_duplicates()
    data.iloc[:, :-1] = data.iloc[:, :-1].apply(zscore)
    data = data.fillna(0).drop_duplicates()
    return shuffle(data)

if __name__ == '__main__':

    print("3 KMeans - Dermatology")
    KMeans(loadData('./data/dermatologyData.csv')).validate()