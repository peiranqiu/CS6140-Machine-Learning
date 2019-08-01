from __future__ import division
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import numpy as np

class KernelDualPerceptron:
	def __init__(self, dataset, RBF = False):
		self.dataset = dataset
		self.RBF = RBF
		self.ep = 10
		self.kf = KFold(n_splits=10, shuffle=True)

	def constantFeature(self, dataset):
		regressionDataset = dataset.copy()
		regressionDataset.columns = range(1, regressionDataset.shape[1] + 1)
		regressionDataset.insert(0, 0, 1)
		return regressionDataset

	def kernel(self, j, i):
		if self.RBF:
			return np.exp(- 0.1 * np.linalg.norm(j - i)**2)
		else:
			return np.dot(j, i.T)

	def rbf(self, j, i):
		return np.exp(- 0.1 * np.linalg.norm(j - i)**2)

	def epoch(self, dataset):
		weight = np.matrix(np.zeros(dataset.shape[0])).T
		epoch = 0
		x = np.matrix(dataset.iloc[:, :-1])
		y = np.matrix(dataset.iloc[:, -1]).T
		n = x.shape[0]

		kernels = [[self.kernel(x[j], x[i]) for j in range(n)] for i in range(n)]

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

	def accuracy(self, trainDataset, testDataset, weight):
		trainX = np.matrix(trainDataset.iloc[:, :-1])
		trainY = np.matrix(trainDataset.iloc[:, -1]).T
		testX = np.matrix(testDataset.iloc[:, :-1])
		testY = np.matrix(testDataset.iloc[:, -1]).T
		Y = []

		for i in range(testDataset.shape[0]):
			sum = 0
			for j in range(trainDataset.shape[0]):
				sum += weight[j] * trainY[j] * self.kernel(trainX[j], testX[i])
			Y.append(1 if sum >= 0.0 else -1)

		return accuracy_score(testY, Y)

	def validate(self):
		epochs = []
		accuracies = []
		fold = 1

		header = ["Fold", "Epoch", "Accuracy"]
		table = []

		for train_index, test_index in self.kf.split(self.dataset):
			print("fold :: {}".format(fold))
			trainDataset = self.constantFeature(self.dataset.iloc[train_index])
			testDataset = self.constantFeature(self.dataset.iloc[test_index])

			epoch, weight = self.epoch(trainDataset)
			epochs.append(epoch)

			accuracy = self.accuracy(trainDataset, testDataset, weight)
			accuracies.append(accuracy)

			row = [fold, epoch, accuracy]
			table.append(row)

			fold += 1

		meanRow = ["Mean", np.mean(epochs), np.mean(accuracies)]
		table.append(meanRow)
		stdRow = ["Standard Deviation", np.std(epochs), np.std(accuracies)]
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))
