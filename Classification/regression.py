from __future__ import division
import Tree
import pandas as pd
import numpy as np

class regression:
	def __init__(self, data, mins):
		self.data = data
		self.mins = mins

	def normalize(self):
		for attribute in self.attributes:
			maxAttribute = self.data[attribute].max()
			minAttribute = self.data[attribute].min()
			self.data[attribute] = self.data[attribute].apply(
				lambda x: (x - minAttribute) / (maxAttribute - minAttribute))

	def splitData(self, data, attribute, val):
		sets = []
		sets.append(data.loc[data[attribute] <= val])
		sets.append(data.loc[data[attribute] > val])
		return sets

	def sumSqrError(self, data):
		error = 0
		for index, row in data.iterrows():
			error += (row[self.targetAttribute] - data[self.targetAttribute].mean()) ** 2
		return error

	def getValues(self, data, attribute):
		vals = set()
		sortedData = data.sort_values(attribute)
		value = sortedData[attribute].iloc[0]
		for index, row in sortedData[1:].iterrows():
			newValue = row[attribute]
			vals.add((value + newValue) / 2)
			value = newValue
		return list(vals)

	def reduceError(self, data, sse, attribute, val):
		error = sse
		for dataset in self.splitData(data, attribute, val):
			error -= self.sumSqrError(dataset) * dataset.shape[0] / data.shape[0]
		return error

	def getReducedError(self, data, sse, attribute):
		vals = self.getValues(data, attribute)
		value = vals[0]
		error = self.reduceError(data, sse, attribute, value)
		for newValue in vals[1:]:
			newError = self.reduceError(data, sse, attribute, newValue)
			if newError < error:
				error = newError
				value = newValue
		return error, value

	def bestAttribute(self, data, sse, attributes):
		best = attributes[0]
		error, value = self.getReducedError(data, sse, best)
		for attribute in attributes[1:]:
			newError, newValue = self.getReducedError(data, sse, attribute)
			if newError > error:
				error = newError
				value = newValue
				best = attribute
		return best, value

	def isPure(self, data):
		return data[self.targetAttribute].unique().shape[0] == 1

	def mostLabel(self, data):
		return data[self.targetAttribute].mean()

	def getChild(self, dataset, threshold, attributes, label):
		if (dataset.shape[0] > 0):
			child = self.buildDecisionTree(dataset, threshold, attributes)
		else:
			child = Tree.BinarySplitTree()
			child.isLeaf = True
			child.label = label
		return child

	def buildDecisionTree(self, data, threshold, attributes):
		root = Tree.BinarySplitTree()
		mostLabel = self.mostLabel(data)
		root.label = mostLabel
		if self.isPure(data) or len(attributes) == 0 or data.shape[0] < threshold:
			root.isLeaf = True
			return root
		else:
			root.isLeaf = False
			root.entropy = self.sumSqrError(data)
			attribute, value = self.bestAttribute(data, root.entropy, attributes)
			root.attribute = attribute
			newAttributes = attributes[:]
			newAttributes.remove(attribute)
			root.value = value
			root.leftChild = self.getChild(data.loc[data[attribute] <= value], threshold, newAttributes, mostLabel)
			root.rightChild = self.getChild(data.loc[data[attribute] > value], threshold, newAttributes, mostLabel)
		return root

	def classify(self, decisionTree, dataset):
		if decisionTree.isLeaf:
			return decisionTree.label
		else:
			value = dataset[decisionTree.attribute]
			if value <= decisionTree.value:
				return self.classify(decisionTree.leftChild, dataset)
			else:
				return self.classify(decisionTree.rightChild, dataset)

	def predict(self, decisionTree, data):
		actuals = []
		predicts = []
		for index, row in data.iterrows():
			actuals.append(row[self.targetAttribute])
			predicts.append(self.classify(decisionTree, row))
		error = 0
		length = len(actuals)
		for i in range(length):
			error += (predicts[i] - actuals[i]) ** 2
		error /= length
		return error ** .5

	def validate(self):
		cols = list(self.data)
		self.attributes = cols[0:len(cols) - 1]
		self.targetAttribute = cols[len(cols) - 1]
		self.normalize()
		rows = self.data.shape[0]
		interval = rows // 10
		sets = []
		start = 0
		end = 0
		shuffledDataSet = self.data.sample(frac=1)
		for i in range(10):
			start = i * interval
			if (i + 2) * interval <= rows:
				end = (i+1) * interval
			else:
				end = rows
			sets.append(shuffledDataSet[start:end])
		for m in self.mins:
			threshold = m * rows
			trainErrors = []
			testErrors = []
			for i in range(10):
				trainData = pd.concat(sets[0:i] + sets[i+1:10])
				testData = sets[i]
				decisionTree = self.buildDecisionTree(trainData, threshold, self.attributes)
				trainError = self.predict(decisionTree, trainData)
				trainErrors.append(trainError)
				testError = self.predict(decisionTree, testData)
				testErrors.append(testError)
			trainSSE = np.mean(trainErrors)
			testSSE = np.mean(testErrors)
			testStd = np.std(testErrors)
			print("MIN: {}, TRAIN SSE: {}, TEST SSE: {}, STANDARD DEVIATION: {}".
				format(m, trainError, testError, testStd))










