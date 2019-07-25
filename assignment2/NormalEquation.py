from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv
import numpy as np

class NormalEquation:

	def __init__(self, dataset):
		self.dataset = dataset

	def normalization(self, dataset, means=[], stds=[]):
		normalizedDataset = dataset.copy()
		attributes = dataset.shape[1] - 1

		if len(means) == attributes:
			for i in range(attributes):
				mean = means[i]
				std = stds[i]
				normalizedDataset[i] = normalizedDataset[i].apply(
					lambda x: (x - mean) / std if std > 0 else 0)

		else:
			means = []
			stds = []
			for i in range(attributes):
				mean = dataset[i].mean()
				means.append(mean)
				std = dataset[i].std()
				stds.append(std)
				normalizedDataset[i] = normalizedDataset[i].apply(
					lambda x: (x - mean) / std if std > 0 else 0)

		return normalizedDataset, means, stds

	def constantFeature(self, dataset):
		regressionDataset = dataset.copy()
		regressionDataset.columns = range(1, regressionDataset.shape[1] + 1)
		regressionDataset.insert(0, 0, 1)
		return regressionDataset

	def normalEquation(self, dataset):
		attributes = dataset.shape[1] - 1
		x = dataset.iloc[:,:-1].values
		y = dataset[attributes]
		return np.dot(np.dot(inv(np.dot(x.transpose(), x)), x.transpose()), y)

	def predict(self, row, w):
		h = 0.0
		attributes = len(row) - 1
		for i in range(attributes):
			h += w[i] * row[i]
		return h - row[attributes]

	def RMSE(self, dataset, w):
		SSE = 0.0
		for index, row in dataset.iterrows():
			SSE += self.predict(row, w)**2
		return SSE, (SSE / dataset.shape[0])**.5

	def validate(self):
		trainSSEs = []
		trainRMSEs = []
		testSSEs = []
		testRMSEs = []
		fold = 1
		print("Fold\tTrain SSE\tTrain RMSE\tTest SSE\tTest RMSE")
		for train_index, test_index in KFold(n_splits=10, shuffle=True).split(self.dataset):
			trainDataset, trainMeans, trainStds = self.normalization(self.dataset.iloc[train_index])
			trainDataset = self.constantFeature(trainDataset)
			w = self.normalEquation(trainDataset)

			trainSSE, trainRMSE = self.RMSE(trainDataset, w)
			trainSSEs.append(trainSSE)
			trainRMSEs.append(trainRMSE)

			testDataset, testMeans, testStds = self.normalization(
				self.dataset.iloc[test_index], trainMeans, trainStds)
			testDataset = self.constantFeature(testDataset)

			testSSE, testRMSE = self.RMSE(testDataset, w)
			testSSEs.append(testSSE)
			testRMSEs.append(testRMSE)

			print("{}\t{}\t{}\t{}\t{}".format(fold, trainSSE, trainRMSE, testSSE, testRMSE))
			fold += 1

		print("{}\t{}\t{}\t{}\t{}".format('Mean', 
			np.mean(trainSSEs), np.mean(trainRMSEs), np.mean(testSSEs), np.mean(testRMSEs)))
		print("{}\t{}\t{}\t{}\t{}".format('Standard Deviation',
			np.std(trainSSEs), np.std(trainRMSEs), np.std(testSSEs), np.std(testRMSEs)))





