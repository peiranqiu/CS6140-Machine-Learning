import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from Multinomial import Multinomial

def loadData(file):
    data = pd.read_csv(file, header=None).fillna(0)
    return data.drop_duplicates()

if __name__ == '__main__':

    print("1 Spambase Logistic Regression")
    LogisticRegression(loadData('./data/spambase.csv')).validate()

    print("1 Breast Cancer Logistic Regression")
    LogisticRegression(loadData('./data/breastcancer.csv')).validate()

    print("1 Diabetes Logistic Regression")
    LogisticRegression(loadData('./data/diabetes.csv')).validate()

    print("2 Multivariate Bernoulli")
    Multinomial(True).validate('./data/20NG_data/train_data.csv', './data/20NG_data/train_label.csv', './data/20NG_data/test_data.csv', './data/20NG_data/test_label.csv')

    print("2 Multinomial")
    Multinomial().validate('./data/20NG_data/train_data.csv', './data/20NG_data/train_label.csv', './data/20NG_data/test_data.csv', './data/20NG_data/test_label.csv')