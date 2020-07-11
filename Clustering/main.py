import pandas as pd
from sklearn.utils import shuffle
from scipy.stats import zscore
from Clustering import Clustering

def loadData(file):
    data = pd.read_csv(file, header=None)
    data = data.fillna(0).drop_duplicates()
    data.iloc[:, :-1] = data.iloc[:, :-1].apply(zscore)
    data = data.fillna(0).drop_duplicates()
    return shuffle(data)

if __name__ == '__main__':

    print("3 KMeans Clustering - Dermatology")
    Clustering(loadData('./data/dermatologyData.csv'), range(1, 13)).validate()

    print("3 KMeans Clustering - Vowels")
    Clustering(loadData('./data/vowelsData.csv'), range(1, 23)).validate()

    print("3 KMeans Clustering - Glass")
    Clustering(loadData('./data/glassData.csv'), range(1, 13)).validate()

    print("3 KMeans Clustering - Ecoli")
    Clustering(loadData('./data/ecoliData.csv'), range(1, 11)).validate()

    print("3 KMeans Clustering - Yeast")
    Clustering(loadData('./data/yeastData.csv'), range(1, 19)).validate()

    print("3 KMeans Clustering - Soybean")
    Clustering(loadData('./data/soybeanData.csv'), range(1, 31)).validate()


    print("4 GMM Clustering - Dermatology")
    Clustering(loadData('./data/dermatologyData.csv'), range(1, 13), True).validate()

    print("4 GMM Clustering - Vowels")
    Clustering(loadData('./data/vowelsData.csv'), range(1, 23), True).validate()

    print("4 GMM Clustering - Glass")
    Clustering(loadData('./data/glassData.csv'), range(1, 13), True).validate()

    print("4 GMM Clustering - Ecoli")
    Clustering(loadData('./data/ecoliData.csv'), range(1, 11), True).validate()

    print("4 GMM Clustering - Yeast")
    Clustering(loadData('./data/yeastData.csv'), range(1, 19), True).validate()

    print("4 GMM Clustering - Soybean")
    Clustering(loadData('./data/soybeanData.csv'), range(1, 31), True).validate()

