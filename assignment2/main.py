import pandas as pd
import numpy as np
from GradientDescent import GradientDescent
from NormalEquation import NormalEquation
from PolynomialRegression import PolynomialRegression
from RidgeRegression import RidgeRegression

def loadData(file):
    data = pd.read_csv(file, header=None)
    return data.drop_duplicates()

if __name__ == '__main__':

    print("2 Housing Gradient Descent")
    #data, learningRate, tolerance
    GradientDescent(loadData('./data/housing.csv').fillna(0), 0.0004, 0.005).validate()

    print("2 Yacht Gradient Descent")
    GradientDescent(loadData('./data/yachtData.csv').fillna(0), 0.001, 0.001).validate()

    print("2 Concrete Gradient Descent")
    GradientDescent(loadData('./data/concreteData.csv').fillna(0), 0.0007, 0.0001).validate()

    print("3 Housing Normal Equation")
    NormalEquation(loadData('./data/housing.csv').fillna(0)).validate()

    print("3 Yacht Normal Equation")
    NormalEquation(loadData('./data/yachtData.csv').fillna(0)).validate()

    print("5 Sinusoid Polynomial Regression")
    #trainData, testData, power
    PolynomialRegression(loadData('./data/sinData_Train.csv').fillna(0), 
        loadData('./data/sinData_Validation.csv').fillna(0), np.arange(1, 16)).validate()

    print("5 Yacht Polynomial Regression")
    PolynomialRegression(loadData('./data/yachtData.csv').fillna(0), None, np.arange(1, 8)).validate()

    print('7 Sinusoid Ridge Regression - 1')
    #data, power, lambda
    RidgeRegression(loadData('./data/sinData_Train.csv').fillna(0), np.arange(1, 6), np.arange(0.0, 10.2, 0.2)).validate()
    
    print('7 Sinusoid Ridge Regression - 2')
    RidgeRegression(loadData('./data/sinData_Train.csv').fillna(0), np.arange(1, 10), np.arange(0.0, 10.2, 0.2)).validate()
