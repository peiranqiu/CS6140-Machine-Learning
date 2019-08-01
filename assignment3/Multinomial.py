from __future__ import division
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
from tabulate import tabulate
import math

class Multinomial:
	def __init__(self,  bernoulli  = False):
		self.bernoulli = bernoulli
		self.pp = {}
		self.cp = {}

	def frequencies(self, dataset):
		frequencies = {}
		for line in open(dataset).readlines():
			doc, word, count = [int(n) for n in line.split()]
			if word in frequencies:
				frequencies[word] += count
			else:
				frequencies[word] = count
		return frequencies

	def train(self, trainData, trainLabel, vocabulary):
		self.pp = {}
		self.cp = {}

		docClasses = [int(n) for n in open(trainLabel).read().split()]
		documents = len(docClasses)
		docFrequency = dict(Counter(docClasses))

		self.pp = dict(map(lambda x: (x[0], math.log(x[1] / documents)), docFrequency.items()))
		frequencies = {}
		classFrequencies = {}

		for line in open(trainData).readlines():
			doc, word, count = [int(n) for n in line.split()]
			if word in vocabulary:
				docClass = docClasses[doc - 1]

				if not self.bernoulli:
					if docClass in classFrequencies:
						classFrequencies[docClass] += count
					else:
						classFrequencies[docClass] = count

				if docClass in frequencies:
					if word not in frequencies[docClass]:
						frequencies[docClass][word] = {}
						frequencies[docClass][word][doc] = count
					else:
						if doc in frequencies[docClass][word]:
							frequencies[docClass][word][doc] += count
						else:
							frequencies[docClass][word][doc] = count
				else:
					frequencies[docClass] = {}
					frequencies[docClass][word] = {}
					frequencies[docClass][word][doc] = count

		for docClass in self.pp.keys():
			self.cp[docClass] = {}
			for word in vocabulary:
				if self.bernoulli:
					num = (len(frequencies.get(docClass, {}).get(word, {}).keys()) + 1) / (docFrequency[docClass] + len(self.pp.keys()))
				else:
					num = (sum(frequencies.get(docClass, {}).get(word, {}).values()) + 1) / (classFrequencies[docClass] + len(vocabulary))
				
				self.cp[docClass][word] = math.log(num)

	def test(self, testData, vocabulary):
		probabilities = {}
		for line in open(testData).readlines():
			doc, word, count = [int(n) for n in line.split()]
			if doc not in probabilities:
				probabilities[doc] = {}
			for docClass in self.pp.keys():
				if docClass not in probabilities[doc]:
					probabilities[doc][docClass] = self.pp[docClass]
				if word in vocabulary:
					probabilities[doc][docClass] += count * self.cp[docClass][word]
		return dict(map(lambda x: (x[0], max(x[1], key=x[1].get)), probabilities.items()))

	def accuracy(self, testLabel, predictions):
		test = [int(n) for n in open(testLabel).read().split()]
		predict = []
		for doc in range(1, len(test) + 1):
			predict.append(predictions[doc])

		return accuracy_score(test, predict), precision_score(test, predict, average="weighted"), recall_score(test, predict, average="weighted")

	def validate(self, trainData, trainLabel, testData, testLabel):
		frequencies = self.frequencies(trainData)
		words = sorted(frequencies, key=frequencies.get, reverse=True)
		table = []

		size = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, len(words)]
		
		for s in size:
			vocabulary = words[:s]
			self.train(trainData, trainLabel, vocabulary)
			predict = self.test(testData, vocabulary)
			accuracy, precision, recall = self.accuracy(testLabel, predict)
			row = [s, accuracy, precision, recall]
			table.append(row)
			if s == len(words):
				test = [int(n) for n in open(testLabel).read().split()]
				prediction = []
				for doc in range(1, len(test) + 1):
					prediction.append(predict[doc])
				_accuracy = accuracy_score(test, prediction)
				_precision = precision_score(test, prediction, average=None)
				_recall = recall_score(test, prediction, average=None)
				
		header = ["Size", "Accuracy", "Precision", "Recall"]
		print(tabulate(table, header, tablefmt='grid'))
		print("Accuracies::\n{}".format(_accuracy))
		print("Precisions::\n{}".format(_precision))
		print("Recalls::\n{}".format(_recall))

