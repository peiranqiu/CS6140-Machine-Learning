from __future__ import division
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import numpy as np

class Perceptron:
	def __init__(self, dataset):
		self.dataset = dataset
		self.learningRate = 0.1
		self.ep = 10
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

	def predict(self, x, weight):
		return 1 if np.dot(x, weight) >= 0.0 else -1

	def epoch(self, dataset):
		attributes = dataset.shape[1] - 1
		weight = np.matrix(np.random.uniform(-1, 1, attributes)).T
		epoch = 0
		x = np.matrix(dataset.iloc[:, :-1])
		y = np.matrix(dataset.iloc[:, -1]).T

		for _ in range(self.ep):
			epoch = _ + 1
			error = 0
			for i in range(x.shape[0]):
				err = y[i] - self.predict(x[i], weight)
				if err != 0:
					weight += (self.learningRate * err * x[i]).T
					error += 1
			if error == 0:
				break
		return epoch, weight

	def accuracy(self, dataset, weight):
		x = np.matrix(dataset.iloc[:, :-1])
		y = (dataset.iloc[:, -1])
		_y = []

		for i in range(dataset.shape[0]):
			_y.append(self.predict(x[i], weight))

		return accuracy_score(y, _y)

	def dualEpoch(self, dataset):
		weight = np.matrix(np.zeros(dataset.shape[0])).T
		epoch = 0
		x = np.matrix(dataset.iloc[:, :-1])
		y = np.matrix(dataset.iloc[:, -1]).T
		n = x.shape[0]

		kernels = [[np.dot(x[j], x[i].T) for j in range(n)] for i in range(n)]

		for _ in range(self.ep):
			epoch = _ + 1
			error = 0
			for i in range(n):
				sum = 0
				for j in range(n):
					sum += weight[j] * y[j] * kernels[i][j]
				_y = 1 if sum >= 0.0 else -1
				if y[i] != _y:
					weight[i] += 1
					error += 1
			if error == 0:
				break

		return epoch, weight

	def dualAccuracy(self, trainDataset, testDataset, weight):
		trainX = np.matrix(trainDataset.iloc[:, :-1])
		trainY = np.matrix(trainDataset.iloc[:, -1]).T
		testX = np.matrix(testDataset.iloc[:, :-1])
		testY = np.matrix(testDataset.iloc[:, -1]).T
		Y = []

		for i in range(testDataset.shape[0]):
			sum = 0
			for j in range(trainDataset.shape[0]):
				sum += weight[j] * trainY[j] * np.dot(trainX[j], testX[i].T)
			Y.append(1 if sum >= 0.0 else -1)

		return accuracy_score(testY, Y)

	def validate(self):
		epochs = []
		accuracies = []
		dualEpochs = []
		dualAccuracies = []
		fold = 1

		header = ["Fold", "Epoch", "Accuracy", "Dual Epoch", "Dual Accuracy"]
		table = []

		for train_index, test_index in self.kf.split(self.dataset):
			trainDataset, trainMeans, trainStds = self.normalization(self.dataset.iloc[train_index])
			trainDataset = self.constantFeature(trainDataset)

			testDataset, testMeans, testStds = self.normalization(
				self.dataset.iloc[test_index], trainMeans, trainStds)
			testDataset = self.constantFeature(testDataset)

			epoch, weight = self.epoch(trainDataset)
			epochs.append(epoch)

			accuracy = self.accuracy(testDataset, weight)
			accuracies.append(accuracy)

			dualEpoch, dualWeight = self.dualEpoch(trainDataset)
			dualEpochs.append(dualEpoch)

			dualAccuracy = self.dualAccuracy(trainDataset, testDataset, dualWeight)
			dualAccuracies.append(dualAccuracy)

			row = [fold, epoch, accuracy, dualEpoch, dualAccuracy]
			table.append(row)

			fold += 1

		meanRow = ["Mean", np.mean(epochs), np.mean(accuracies), np.mean(dualEpochs), np.mean(dualAccuracies)]
		table.append(meanRow)
		stdRow = ["Standard Deviation", np.std(epochs), np.std(accuracies), np.std(dualEpochs), np.std(dualAccuracies)]
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))






