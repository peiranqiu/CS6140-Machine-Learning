from __future__ import division
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.svm import SVC
from tabulate import tabulate
import numpy as np

class Multiclass:
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
			trainX = trainDataset.iloc[:, 1:]
			trainY = trainDataset.iloc[:, 0].T
			_trainDatasets = []
			_testDatasets = []
			accuracy  = [-1] * 3
			cs.append([0.0] * 3)
			if self.RBF:
				gs.append([0.0] * 3)

			for _trainIndex, _testIndex in self.mf.split(trainDataset):
				_trainDatasets.append(trainDataset.iloc[_trainIndex])
				_testDatasets.append(trainDataset.iloc[_testIndex])

			for _c in self.cList:
				if self.RBF:
					for _g in self.gList:
						_accuracy = 0.0
						for i in range(self.m):
							_trainDataset = _trainDatasets[i]
							_trainX = _trainDataset.iloc[:, 1:]
							_trainY = _trainDataset.iloc[:, 0].T
							_clf = OneVsRestClassifier(SVC(C=_c, gamma=_g))
							_clf.fit(_trainX, _trainY)

							_testDataset = _testDatasets[i]
							_testX = _testDataset.iloc[:, 1:]
							_testY = _testDataset.iloc[:, 0].T
							cmat = confusion_matrix(_testY, _clf.predict(_testX))
							_accuracy += cmat.diagonal()/cmat.sum(axis=1)

						for i in range(3):
							if _accuracy[i] >= accuracy[i]:
								accuracy[i] = _accuracy[i]
								cs[-1][i] = _c
								gs[-1][i] = _g

				else:
					_accuracy = 0.0
					for i in range(self.m):
						_trainDataset = _trainDatasets[i]
						_trainX = _trainDataset.iloc[:, 1:]
						_trainY = _trainDataset.iloc[:, 0].T
						_clf = OneVsRestClassifier(SVC(kernel='linear', C=_c))
						_clf.fit(_trainX, _trainY)

						_testDataset = _testDatasets[i]
						_testX = _testDataset.iloc[:, 1:]
						_testY = _testDataset.iloc[:, 0].T
						cmat = confusion_matrix(_testY, _clf.predict(_testX))
						_accuracy += cmat.diagonal()/cmat.sum(axis=1)

					for i in range(3):
						if _accuracy[i] >= accuracy[i]:
							accuracy[i] = _accuracy[i]
							cs[-1][i] = _c

			testDataset = self.dataset.iloc[testIndex]
			testX = testDataset.iloc[:, 1:]
			testY = testDataset.iloc[:, 0].T

			trainAccuracies.append([0.0] * 3)
			trainPrecisions.append([0.0] * 3)
			trainRecalls.append([0.0] * 3)
			testAccuracies.append([0.0] * 3)
			testPrecisions.append([0.0] * 3)
			testRecalls.append([0.0] * 3)

			for i in range(3):
				if self.RBF:
					clf = OneVsRestClassifier(SVC(C=cs[-1][i], gamma=gs[-1][i]))
				else:
					clf = OneVsRestClassifier(SVC(kernel='linear', C=cs[-1][i]))
				clf.fit(trainX, trainY)

				_train = clf.predict(trainX)
				trainCmat = confusion_matrix(trainY, _train)
				trainAccuracy = trainCmat.diagonal() / trainCmat.sum(axis=1)
				trainScore = precision_recall_fscore_support(trainY, _train)
				trainPrecision = trainScore[0]
				trainRecall = trainScore[1]
				trainAccuracies[-1][i] = trainAccuracy[i]
				trainPrecisions[-1][i] = trainPrecision[i]
				trainRecalls[-1][i] = trainRecall[i]

				_test = clf.predict(testX)
				testCmat = confusion_matrix(testY, _test)
				testAccuracy = testCmat.diagonal() / testCmat.sum(axis=1)
				testScore = precision_recall_fscore_support(testY, _test)
				testPrecision = testScore[0]
				testRecall = testScore[1]
				testAccuracies[-1][i] = testAccuracy[i]
				testPrecisions[-1][i] = testPrecision[i]
				testRecalls[-1][i] = testRecall[i]

			row = [fold, cs[-1], trainAccuracies[-1], trainPrecisions[-1], trainRecalls[-1], testAccuracies[-1], testPrecisions[-1], testRecalls[-1]]
			if self.RBF:
				row.insert(2, gs[-1])
			table.append(row)

			fold += 1

		header = ["Fold", "C1 C2 C3", "Train Accuracy 1 2 3", "Train Precision 1 2 3", "Train Recall 1 2 3", "Test Accuracy 1 2 3", "Test Precision 1 2 3", "Test Recall 1 2 3"]
		meanRow = ["Mean", np.mean(cs, axis=0), np.mean(trainAccuracies, axis=0), np.mean(trainPrecisions, axis=0), np.mean(trainRecalls, axis=0), np.mean(testAccuracies, axis=0), np.mean(testPrecisions, axis=0), np.mean(testRecalls, axis=0)]
		stdRow = ["Standard Deviation", np.std(cs, axis=0), np.std(trainAccuracies, axis=0), np.std(trainPrecisions, axis=0), np.std(trainRecalls, axis=0), np.std(testAccuracies, axis=0), np.std(testPrecisions, axis=0), np.std(testRecalls, axis=0)]
		if self.RBF:
			header.insert(2, "G1 G2 G3")
			meanRow.insert(2, np.mean(gs, axis=0))
			stdRow.insert(2, np.std(gs, axis=0))
		table.append(meanRow)
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))
			

	def AUC(self):
		accuracies = []
		cs = []
		gs = []
		fprs = []
		tprs = []
		fold = 1
		table = []

		for trainIndex, testIndex in self.kf.split(self.dataset):
			trainDataset = self.dataset.iloc[trainIndex]
			trainX = trainDataset.iloc[:, 1:]
			trainY = label_binarize(trainDataset.iloc[:, 0].T, classes=[1, 2, 3])
			_trainDatasets = []
			_testDatasets = []
			accuracy  = [-1] * 3
			cs.append([0.0] * 3)

			if self.RBF:
				gs.append([0.0] * 3)

			for _trainIndex, _testIndex in self.mf.split(trainDataset):
				_trainDatasets.append(trainDataset.iloc[_trainIndex])
				_testDatasets.append(trainDataset.iloc[_testIndex])

			for _c in self.cList:
				if self.RBF:
					for _g in self.gList:
						_accuracy = [0.0] * 3
						for i in range(self.m):
							_trainDataset = _trainDatasets[i]
							_trainX = _trainDataset.iloc[:, 1:]
							_trainY = label_binarize(_trainDataset.iloc[:, 0].T, classes=[1, 2, 3])
							_clf = OneVsRestClassifier(SVC(C=_c, gamma=_g, probability=True))
							_clf.fit(_trainX, _trainY)

							_testDataset = _testDatasets[i]
							_testX = _testDataset.iloc[:, 1:]
							_testY = label_binarize(_testDataset.iloc[:, 0].T, classes=[1, 2, 3])
							_proba = _clf.predict_proba(_testX)
							for j in range(3):
								_fpr, _tpr, _thre = metrics.roc_curve(_testY[:, j], _proba[:, j])
								_accuracy[j] += metrics.auc(_fpr, _tpr)

						for i in range(3):
							if _accuracy[i] >= accuracy[i]:
								accuracy[i] = _accuracy[i]
								cs[-1][i] = _c
								gs[-1][i] = _g

				else:
					_accuracy = [0.0] * 3
					for i in range(self.m):
						_trainDataset = _trainDatasets[i]
						_trainX = _trainDataset.iloc[:, 1:]
						_trainY = label_binarize(_trainDataset.iloc[:, 0].T, classes=[1, 2, 3])
						_clf = OneVsRestClassifier(SVC(kernel='linear', C=_c, probability=True))
						_clf.fit(_trainX, _trainY)

						_testDataset = _testDatasets[i]
						_testX = _testDataset.iloc[:, 1:]
						_testY = label_binarize(_testDataset.iloc[:, 0].T, classes=[1, 2, 3])
						_proba = _clf.predict_proba(_testX)
						for j in range(3):
							_fpr, _tpr, _thre = metrics.roc_curve(_testY[:, j], _proba[:, j])
							_accuracy[j] += metrics.auc(_fpr, _tpr)

					for i in range(3):
						if _accuracy[i] >= accuracy[i]:
							accuracy[i] = _accuracy[i]
							cs[-1][i] = _c

			testDataset = self.dataset.iloc[testIndex]
			testX = testDataset.iloc[:, 1:]
			testY = label_binarize(testDataset.iloc[:, 0].T, classes=[1, 2, 3]) 

			accuracies.append([0.0] * 3)
			_fprs = []
			_tprs = []

			for i in range(3):
				if self.RBF:
					clf = OneVsRestClassifier(SVC(C=cs[-1][i], gamma=gs[-1][i], probability=True))
				else:
					clf = OneVsRestClassifier(SVC(kernel='linear', C=cs[-1][i], probability=True))
				clf.fit(trainX, trainY)

				proba = clf.predict_proba(testX)
				fpr, tpr, thre = metrics.roc_curve(testY[:, i], proba[:, i])
				_fprs.append(fpr)
				_tprs.append(tpr)
				accuracies[-1][i] = metrics.auc(fpr, tpr)

			fprs.append(_fprs)
			tprs.append(_tprs)

			row = [fold, cs[-1], accuracies[-1]]
			if self.RBF:
				row.insert(2, gs[-1])
			table.append(row)

			fold += 1

		header = ["Fold", "C1 C2 C3", "Test Accuracy 1 2 3"]
		meanRow = ["Mean", np.mean(cs, axis=0), np.mean(accuracies, axis=0)]
		stdRow = ["Standard Deviation", np.std(cs, axis=0), np.std(accuracies, axis=0)]
		if self.RBF:
			header.insert(2, "G1 G2 G3")
			meanRow.insert(2, np.mean(gs, axis=0))
			stdRow.insert(2, np.std(gs, axis=0))
		table.append(meanRow)
		table.append(stdRow)
		print(tabulate(table, header, tablefmt='grid'))




		















		