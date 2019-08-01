from __future__ import division
from sklearn.model_selection import KFold
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

class LogisticRegression:
	def __init__(self, dataset):
		self.dataset = dataset
		self.learningRate = 0.75
		self.tolerance  = 0.00001
		self.iterations = 1000
		self.kf = KFold(n_splits=10, shuffle=True)

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

	def probability(self, x, th):
		return 1.0 / (1.0 + np.exp(-np.dot(x, th)))

	def cost(self, x, y, th):
		p = self.probability(x, th)
		return (np.multiply(-y, np.log(p)) - np.multiply((1 - y), np.log(1 - p))).mean()

	def gradient(self, x, y, th):
		p = self.probability(x, th)
		return np.dot(x.T, p - y) / x.shape[0]

	def logisticRegression(self, dataset, plot):
		x = np.matrix(dataset.iloc[:, :-1])
		y = np.matrix(dataset.iloc[:, -1]).T
		th = np.matrix(np.zeros(dataset.shape[1] - 1)).T
		logistic = []
		logistic.append(self.cost(x, y, th))

		for i in range(self.iterations):
			_th = th - self.learningRate * self.gradient(x, y, th)
			_cost = self.cost(x, y, _th)
			if(logistic[i] - _cost > self.tolerance):
				logistic.append(_cost)
				th = _th
			else:
				break

		if plot:
			plt.plot(logistic, label='Logistic Regression')	
			plt.xlabel('Iteration')
			plt.ylabel('Logistic')
			plt.show()

		return th

	def predict(self, x, th):
		probabilities = self.probability(x, th)
		return [0 if p < 0.5 else 1 for p in probabilities]

	def accuracy(self, dataset, th):
		x = np.matrix(dataset.iloc[:, :-1])
		y = dataset.iloc[:, -1].values.tolist()
		_y = self.predict(x, th)
		tp = 0

		for index, value in enumerate(_y):
			if value == 1 and y[index] == 1:
				tp += 1

		accuracy = (_y == dataset.iloc[:, -1]).mean()
		precision = tp / _y.count(1)
		recall = tp / y.count(1)

		return accuracy, precision, recall

	def validate(self):

		trainAccuracies = []
		trainPrecisions = []
		trainRecalls = []
		testAccuracies = []
		testPrecisions = []
		testRecalls = []
		fold = 1
		plotFold = random.randint(1, 10)
		table = []

		for train_index, test_index in self.kf.split(self.dataset):
			trainDataset, trainMeans, trainStds = self.normalization(self.dataset.iloc[train_index])
			trainDataset = self.constantFeature(trainDataset)
			testDataset, testMeans, testStds = self.normalization(
				self.dataset.iloc[test_index], trainMeans, trainStds)
			testDataset = self.constantFeature(testDataset)

			th = self.logisticRegression(trainDataset, fold == plotFold)
			trainAccuracy, trainPrecision, trainRecall = self.accuracy(trainDataset, th)
			testAccuracy, testPrecision, testRecall = self.accuracy(testDataset, th)
			
			trainAccuracies.append(trainAccuracy)
			trainPrecisions.append(trainPrecision)
			trainRecalls.append(trainRecall)
			testAccuracies.append(testAccuracy)
			testPrecisions.append(testPrecision)
			testRecalls.append(testRecall)

			row = [fold, trainAccuracy, trainPrecision, trainRecall, trainAccuracy, trainPrecision, trainRecall]
			table.append(row)

			fold += 1

		header = ["Fold", "Train Accuracy", "Train Precision", "Train Recall", "Test Accuracy", "Test Precision", "Test Recall"]
		meanRow = ["Mean", np.mean(trainAccuracies), np.mean(trainPrecisions), np.mean(trainRecalls), np.mean(testAccuracies), np.mean(testPrecisions), np.mean(testRecalls)]
		table.append(meanRow)
		stdRow = ["Standard Deviation", np.std(trainAccuracies), np.std(trainPrecisions), np.std(trainRecalls), np.std(testAccuracies), np.std(testPrecisions), np.std(testRecalls)]
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))








