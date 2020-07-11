from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RidgeRegression:
	def __init__(self, dataset, p, l):
		self.dataset = dataset
		self.p = p
		self.l = l

	def powerDataset(self, dataset, p):
		columns = dataset.shape[1] - 1
		powerDataset = pd.DataFrame(index=dataset.index)
		for i in range(1, p+1):
			powerDataset[np.arange(columns * (i - 1), columns * i)] = np.power(dataset[range(columns)], i)
		powerDataset[columns * p] = dataset[columns]
		return powerDataset

	def normalization(self, dataset, p, means=[]):
		normalizedDataset = self.powerDataset(dataset, p)
		attributes = normalizedDataset.shape[1] - 1

		if len(means) == attributes:
			for i in range(attributes):
				mean = means[i]
				normalizedDataset[i] = normalizedDataset[i].apply(lambda x: x - mean)

		else:
			means = []
			for i in range(attributes):
				mean = normalizedDataset[i].mean()
				means.append(mean)
				normalizedDataset[i] = normalizedDataset[i].apply(lambda x: x - mean)

		return normalizedDataset, means

	def ridgeRegression(self, dataset, l):
		attributes = dataset.shape[1] - 1
		x = dataset.iloc[:,:-1].values
		y = dataset[attributes]
		w = np.insert(
			np.dot(np.dot(inv(np.dot(x.transpose(), x)), x.transpose()), y), 
			0, y.mean())
		return w

	def predict(self, row, w):
		h = w[0]
		attributes = len(row) - 1
		for i in range(attributes):
			h += w[i+1] * row[i]
		return h - row[attributes]

	def RMSE(self, dataset, w):
		SSE = 0.0
		for index, row in dataset.iterrows():
			SSE += self.predict(row, w)**2
		return SSE, (SSE / dataset.shape[0])**.5

	def validate(self):
		for p in self.p:
			print("p :: {}".format(p))
			_trainRMSEs = []
			_testRMSEs = []

			for l in self.l:
				print("λ :: {}".format(l))
				trainSSEs = []
				trainRMSEs = []
				testSSEs = []
				testRMSEs = []
				fold = 1
				print("Fold\tTrain SSE\tTrain RMSE\tTest SSE\tTest RMSE")
				for train_index, test_index in KFold(n_splits=10, shuffle=True).split(self.dataset):
					trainDataset, trainMeans= self.normalization(self.dataset.iloc[train_index], p)
					w = self.ridgeRegression(trainDataset, l)

					trainSSE, trainRMSE = self.RMSE(trainDataset, w)
					trainSSEs.append(trainSSE)
					trainRMSEs.append(trainRMSE)

					testDataset, testMeans = self.normalization(
						self.dataset.iloc[test_index], p, trainMeans)
					testSSE, testRMSE = self.RMSE(testDataset, w)
					testSSEs.append(testSSE)
					testRMSEs.append(testRMSE)
					
					print("{}\t{}\t{}\t{}\t{}".format(fold, trainSSE, trainRMSE, testSSE, testRMSE))
					fold += 1

				print("{}\t{}\t{}\t{}\t{}".format('Mean', 
					np.mean(trainSSEs), np.mean(trainRMSEs), np.mean(testSSEs), np.mean(testRMSEs)))
				print("{}\t{}\t{}\t{}\t{}".format('Standard Deviation',
					np.std(trainSSEs), np.std(trainRMSEs), np.std(testSSEs), np.std(testRMSEs)))

				_trainRMSEs.append(np.mean(trainRMSEs))
				_testRMSEs.append(np.mean(testRMSEs))

			plt.plot(self.l, _trainRMSEs, label='Train Dataset')
			plt.plot(self.l, _testRMSEs, label='Test Dataset')
			plt.xlabel('λ')
			plt.ylabel('Mean RMSE')
			plt.title('Ridge Regression p={}'.format(p))
			plt.legend()
			plt.show()
