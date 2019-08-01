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
		self.tolerance  = 0.0000001
		self.b = 0.05
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
		iteration = 0
		logistic.append(self.cost(x, y, th))

		for i in range(self.iterations):
			iteration = i + 1
			_th = th - self.learningRate * self.gradient(x, y, th)
			_cost = self.cost(x, y, _th)
			if(logistic[i] - _cost > self.tolerance):
				logistic.append(_cost)
				th = _th
			else:
				break

		if plot:
			plt.plot(logistic, label='Logistic Regression')

		return th, iteration

	def regularizeCost(self, x, y, th):
		p = self.probability(x, th)
		_th = th.copy()
		_th[0] = 0
		return (np.multiply(-y, np.log(p)) - np.multiply((1 - y), np.log(1 - p))).mean() + (self.b / (2 * x.shape[0])) * np.sum(np.square(_th))

	def regularizeGradient(self, x, y, th):
		p = self.probability(x, th)
		return np.dot(x.T, p - y) / x.shape[0]

	def regularizeLogisticRegression(self, dataset, plot):
		x = np.matrix(dataset.iloc[:, :-1])
		y = np.matrix(dataset.iloc[:, -1]).T
		th = np.matrix(np.zeros(dataset.shape[1] - 1)).T
		logistic = []
		iteration = 0
		logistic.append(self.regularizeCost(x, y, th))

		for i in range(self.iterations):
			iteration = i + 1
			theta = th.copy()
			theta[0] = 0
			_th = th - self.learningRate * self.regularizeGradient(x, y, th) - (self.b / x.shape[0]) * theta
			_cost = self.regularizeCost(x, y, _th)
			if(logistic[i] - _cost > self.tolerance):
				logistic.append(_cost)
				th = _th
			else:
				break

		if plot:
			plt.plot(logistic, label='Regularized Logistic Regression')

		return th, iteration

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
		iterations = []
		accuracies = []
		precisions = []
		recalls = []
		_iterations = []
		_accuracies = []
		_precisions = []
		_recalls = []
		fold = 1
		plotFold = random.randint(1, 10)
		header = ["Fold", "Iteration", "Accuracy", "Precision", "Recall", "Regularized Iteration", "Regularized Accuracy", "Regularized Precision", "Regularized Recall"]
		table = []

		plt.figure()

		for train_index, test_index in self.kf.split(self.dataset):
			trainDataset, trainMeans, trainStds = self.normalization(self.dataset.iloc[train_index])
			trainDataset = self.constantFeature(trainDataset)

			testDataset, testMeans, testStds = self.normalization(
				self.dataset.iloc[test_index], trainMeans, trainStds)
			testDataset = self.constantFeature(testDataset)

			th, iteration = self.logisticRegression(trainDataset, fold == plotFold)
			iterations.append(iteration)

			accuracy, precision, recall = self.accuracy(testDataset, th)
			accuracies.append(accuracy)
			precisions.append(precision)
			recalls.append(recall)

			_th, _iteration = self.regularizeLogisticRegression(trainDataset, fold == plotFold)
			_iterations.append(_iteration)

			_accuracy, _precision, _recall = self.accuracy(testDataset, _th)
			_accuracies.append(_accuracy)
			_precisions.append(_precision)
			_recalls.append(_recall)

			row = [fold, iteration, accuracy, precision, recall, _iteration, _accuracy, _precision, _recall]
			table.append(row)

			fold += 1

		meanRow = ["Mean", np.mean(iterations), np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(_iterations), np.mean(_accuracies), np.mean(_precisions), np.mean(_recalls)]
		table.append(meanRow)
		stdRow = ["Standard Deviation", np.std(iterations), np.std(accuracies), np.std(precisions), np.std(recalls), np.std(_iterations), np.std(_accuracies), np.std(_precisions), np.std(_recalls)]
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))

		plt.xlabel('Iteration')
		plt.ylabel('Logistic')
		plt.title('Regularized Logistic Regression')
		plt.show()












