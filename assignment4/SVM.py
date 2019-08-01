from __future__ import division
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class SVM:
	def __init__(self, dataset, RBF = False):
		self.dataset = dataset
		self.RBF = RBF
		self.cList = [2**i for i in range(-5, 11)]
		self.gList = [2**i for i in range(-15, 6)]
		self.m = 5
		self.mf = KFold(n_splits=5, shuffle=True)
		self.kf = KFold(n_splits=10, shuffle=True)

	def SVM(self):
		trainAccuracies = []
		trainPrecisions = []
		trainRecalls = []
		testAccuracies = []
		testPrecisions = []
		testRecalls = []
		cs = []
		gs = []
		fold = 1
		table = []

		for trainIndex, testIndex in self.kf.split(self.dataset):
			trainDataset = self.dataset.iloc[trainIndex]
			trainX = trainDataset.iloc[:, :-1]
			trainY = trainDataset.iloc[:, -1].T
			_trainDatasets = []
			_testDatasets = []
			accuracy  = -1
			c = 0.0
			g = 0.0

			for _trainIndex, _testIndex in self.mf.split(trainDataset):
				_trainDatasets.append(trainDataset.iloc[_trainIndex])
				_testDatasets.append(trainDataset.iloc[_testIndex])

			for _c in self.cList:
				if self.RBF:
					for _g in self.gList:
						_accuracy = 0.0
						for i in range(self.m):
							_trainDataset = _trainDatasets[i]
							_trainX = _trainDataset.iloc[:, :-1]
							_trainY = _trainDataset.iloc[:, -1].T
							_clf = SVC(C=_c, gamma=_g)
							_clf.fit(_trainX, _trainY)

							_testDataset = _testDatasets[i]
							_testX = _testDataset.iloc[:, :-1]
							_testY = _testDataset.iloc[:, -1].T
							_accuracy += accuracy_score(_testY, _clf.predict(_testX))

						_accuracy /= self.m
						if(_accuracy >= accuracy):
							accuracy = _accuracy
							c = _c
							g = _g

				else:
					_accuracy = 0.0
					for i in range(self.m):
						_trainDataset = _trainDatasets[i]
						_trainX = _trainDataset.iloc[:, :-1]
						_trainY = _trainDataset.iloc[:, -1].T
						_clf = SVC(kernel='linear', C=_c)
						_clf.fit(_trainX, _trainY)

						_testDataset = _testDatasets[i]
						_testX = _testDataset.iloc[:, :-1]
						_testY = _testDataset.iloc[:, -1].T
						_accuracy += accuracy_score(_testY, _clf.predict(_testX))

					_accuracy /= self.m
					if(_accuracy >= accuracy):
						accuracy = _accuracy
						c = _c

			cs.append(c)
			if self.RBF:
				gs.append(g)
				clf = SVC(C=c, gamma=g)
			else:
				clf = SVC(kernel='linear', C=c)

			clf.fit(trainX, trainY)
			_train = clf.predict(trainX)
			trainAccuracy = accuracy_score(trainY, _train)
			trainPrecision = precision_score(trainY, _train)
			trainRecall = recall_score(trainY, _train)

			trainAccuracies.append(trainAccuracy)
			trainPrecisions.append(trainPrecision)
			trainRecalls.append(trainRecall)

			testDataset = self.dataset.iloc[testIndex]
			testX = testDataset.iloc[:, :-1]
			testY = testDataset.iloc[:, -1].T
			_test = clf.predict(testX)
			testAccuracy = accuracy_score(testY, _test)
			testPrecision = precision_score(testY, _test)
			testRecall = recall_score(testY, _test)

			testAccuracies.append(testAccuracy)
			testPrecisions.append(testPrecision)
			testRecalls.append(testRecall)

			row = [fold, c, trainAccuracy, trainPrecision, trainRecall, testAccuracy, testPrecision, testRecall]
			if self.RBF:
				row.insert(2, g)
			table.append(row)

			fold += 1

		header = ["Fold", "C", "Train Accuracy", "Train Precision", "Train Recall", "Test Accuracy", "Test Precision", "Test Recall"]
		meanRow = ["Mean", np.mean(cs), np.mean(trainAccuracies), np.mean(trainPrecisions), np.mean(trainRecalls), np.mean(testAccuracies), np.mean(testPrecisions), np.mean(testRecalls)]
		stdRow = ["Standard Deviation", np.std(cs), np.std(trainAccuracies), np.std(trainPrecisions), np.std(trainRecalls), np.std(testAccuracies), np.std(testPrecisions), np.std(testRecalls)]
		if self.RBF:
			header.insert(2, "G")
			meanRow.insert(2, np.mean(gs))
			stdRow.insert(2, np.std(gs))
		table.append(meanRow)
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))


	def AUC(self):
		accuracies = []
		cs = []
		gs = []
		fold = 1
		table = []
		plt.figure()

		for trainIndex, testIndex in self.kf.split(self.dataset):
			trainDataset = self.dataset.iloc[trainIndex]
			trainX = trainDataset.iloc[:, :-1]
			trainY = trainDataset.iloc[:, -1].T
			_trainDatasets = []
			_testDatasets = []
			accuracy  = -1
			c = 0.0
			g = 0.0

			for _trainIndex, _testIndex in self.mf.split(trainDataset):
				_trainDatasets.append(trainDataset.iloc[_trainIndex])
				_testDatasets.append(trainDataset.iloc[_testIndex])

			for _c in self.cList:
				if self.RBF:
					for _g in self.gList:
						_accuracy = 0.0
						for i in range(self.m):
							_trainDataset = _trainDatasets[i]
							_trainX = _trainDataset.iloc[:, :-1]
							_trainY = _trainDataset.iloc[:, -1].T
							_clf = SVC(C=_c, gamma=_g, probability=True)
							_clf.fit(_trainX, _trainY)

							_testDataset = _testDatasets[i]
							_testX = _testDataset.iloc[:, :-1]
							_testY = _testDataset.iloc[:, -1].T

							_proba = _clf.predict_proba(_testX)
							_fpr, _tpr, _thre = metrics.roc_curve(_testY, _proba[:, 1])
							_accuracy += metrics.auc(_fpr, _tpr)

						_accuracy /= self.m
						if(_accuracy >= accuracy):
							accuracy = _accuracy
							c = _c
							g = _g

				else:
					_accuracy = 0.0
					for i in range(self.m):
						_trainDataset = _trainDatasets[i]
						_trainX = _trainDataset.iloc[:, :-1]
						_trainY = _trainDataset.iloc[:, -1].T
						_clf = SVC(kernel='linear', C=_c, probability=True)
						_clf.fit(_trainX, _trainY)

						_testDataset = _testDatasets[i]
						_testX = _testDataset.iloc[:, :-1]
						_testY = _testDataset.iloc[:, -1].T

						_proba = _clf.predict_proba(_testX)
						_fpr, _tpr, _thre = metrics.roc_curve(_testY, _proba[:, 1])
						_accuracy += metrics.auc(_fpr, _tpr)

					_accuracy /= self.m
					if(_accuracy >= accuracy):
						accuracy = _accuracy
						c = _c

			cs.append(c)
			if self.RBF:
				gs.append(g)
				clf = SVC(C=c, gamma=g, probability=True)
			else:
				clf = SVC(kernel='linear', C=c, probability=True)

			clf.fit(trainX, trainY)
			
			testDataset = self.dataset.iloc[testIndex]
			testX = testDataset.iloc[:, :-1]
			testY = testDataset.iloc[:, -1].T

			proba = clf.predict_proba(testX)
			fpr, tpr, thre = metrics.roc_curve(testY, proba[:, 1])
			accuracy = metrics.auc(fpr, tpr)
			accuracies.append(accuracy)
			plt.plot(fpr, tpr, lw=2, label='Fold %d AUC = %0.3f' % (fold, accuracy))

			row = [fold, c, accuracy]
			if self.RBF:
				row.insert(2, g)
			table.append(row)

			fold += 1

		header = ["Fold", "C", "Accuracy"]
		meanRow = ["Mean", np.mean(cs), np.mean(accuracies)]
		stdRow = ["Standard Deviation", np.std(cs), np.std(accuracies)]
		if self.RBF:
			header.insert(2, "G")
			meanRow.insert(2, np.mean(gs))
			stdRow.insert(2, np.std(gs))
		table.append(meanRow)
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))

		plt.plot([0, 1], [0, 1], 'r--', lw=2)
		plt.xlabel('FP')
		plt.ylabel('TP')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.title('ROC')
		plt.legend()
		plt.show()

