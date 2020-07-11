import pandas as pd
from sklearn.utils import shuffle
from Perceptron import Perceptron
from KernelDualPerceptron import KernelDualPerceptron
from LogisticRegression import LogisticRegression
from SVM import SVM
from Multiclass import Multiclass

def loadData(file):
    data = pd.read_csv(file, header=None)
    data = data.fillna(0).drop_duplicates()
    return shuffle(data)

if __name__ == '__main__':

    print("1.2 Perceptron - PerceptronData")
    Perceptron(loadData('./data/perceptronData.csv')).validate()

    print("1.3 Linear Kernel - Two Spiral")
    KernelDualPerceptron(loadData('./data/twoSpirals.csv')).validate()

    print("1.3 Gaussian Kernel - Two Spiral")
    KernelDualPerceptron(loadData('./data/twoSpirals.csv'), True).validate()

    print("2 Logistic Regression - Diabetes")
    LogisticRegression(loadData('./data/diabetes.csv')).validate()

    print("2 Logistic Regression - Breast Cancer")
    LogisticRegression(loadData('./data/breastcancer.csv')).validate()

    print("2 Logistic Regression - Spambase")
    LogisticRegression(loadData('./data/spambase.csv')).validate()

    print("3 SVM Linear - Diabetes")
    SVM(loadData('./data/diabetes.csv')).SVM()

    print("3 SVM RBF - Diabetes")
    SVM(loadData('./data/diabetes.csv'), True).SVM()

    print("3 SVM Linear AUC - Diabetes")
    SVM(loadData('./data/diabetes.csv')).AUC()

    print("3 SVM RBF AUC - Diabetes")
    SVM(loadData('./data/diabetes.csv'), True).AUC()

    print("3 SVM Linear - Breast Cancer")
    SVM(loadData('./data/breastcancer.csv')).SVM()

    print("3 SVM RBF - Breast Cancer")
    SVM(loadData('./data/breastcancer.csv'), True).SVM()

    print("3 SVM Linear AUC - Breast Cancer")
    SVM(loadData('./data/breastcancer.csv')).AUC()

    print("3 SVM RBF AUC - Breast Cancer")
    SVM(loadData('./data/breastcancer.csv'), True).AUC()

    print("3 SVM Linear - Spambase")
    SVM(loadData('./data/spambase.csv')).SVM()

    print("3 SVM RBF - Spambase")
    SVM(loadData('./data/spambase.csv'), True).SVM()

    print("3 SVM Linear AUC - Spambase")
    SVM(loadData('./data/spambase.csv')).AUC()

    print("3 SVM RBF AUC - Spambase")
    SVM(loadData('./data/spambase.csv'), True).AUC()

    print("4 Multiclass Linear - Wine")
    Multiclass(loadData('./data/wine.csv')).SVM()

    print("4 Multiclass RBF - Wine")
    Multiclass(loadData('./data/wine.csv'), True).SVM()

    print("4 Multiclass Linear AUC - Wine")
    Multiclass(loadData('./data/wine.csv')).AUC()

    print("4 Multiclass RBF AUC - Wine")
    Multiclass(loadData('./data/wine.csv'), True).AUC()
 