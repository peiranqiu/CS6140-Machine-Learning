from __future__ import division
import Tree
import pandas as pd
import numpy as np
from pandas_ml import ConfusionMatrix

class decisionTree:
	def __init__(self, data, mins, binary):
		self.data = data
		self.mins = mins
		self.binary = binary

	def normalize(self):
		if self.binary:
			for attribute in self.attributes:
				maxAttribute = self.data[attribute].max()
				minAttribute = self.data[attribute].min()
				self.data[attribute] = self.data[attribute].apply(
					lambda x: (x - minAttribute) / (maxAttribute - minAttribute))

	def entropy(self, data):
		splits = data[self.targetAttribute].value_counts().to_dict()
		entropy = 0
		for label, count in splits.items():
			freq = count / data.shape[0]
			entropy = entropy - freq * np.log2(freq)
		return entropy

	def getValues(self, data, attribute):
		vals = set()
		sortedData = data.sort_values(attribute)
		value = sortedData[attribute].iloc[0]
		label = sortedData[self.targetAttribute].iloc[0]
		for index, row in sortedData[1:].iterrows():
			newLabel = row[self.targetAttribute]
			newValue = row[attribute]
			if newLabel != label:
				vals.add((value + newValue)/2)
				label = newLabel
			value = newValue
		return list(vals)

	def splitData(self, data, attribute, val = None):
		sets = []
		if self.binary:
			sets.append(data.loc[data[attribute] < val])
			sets.append(data.loc[data[attribute] >= val])
		else:
			for value in data[attribute].unique():
				sets.append(data.loc[data[attribute] == value])
		return sets

	def informationGain(self, data, entropy, attribute, val = None):
		ig = entropy
		for dataset in self.splitData(data, attribute, val):
			ig = ig - (dataset.shape[0] / data.shape[0]) * self.entropy(dataset)
		return ig

	def getInformationGain(self, data, entropy, attribute):
		if self.binary:
			vals = self.getValues(data, attribute)
			value = vals[0]
			ig = self.informationGain(data, entropy, attribute, value)
			for newValue in vals[1:]:
				newIg = self.informationGain(data, entropy, attribute, newValue)
				if(newIg > ig):
					ig = newIg
					value = newValue
			return ig, value
		else:
			return self.informationGain(data, entropy, attribute), 0

	def bestAttribute(self, data, entropy, attributes):
		best = attributes[0]
		ig, value = self.getInformationGain(data, entropy, best)
		for attribute in attributes[1:]:
			newIg, newValue = self.getInformationGain(data, entropy, attribute)
			if newIg > ig:
				ig = newIg
				value = newValue
				best = attribute
		return best, value

	def isPure(self, data):
		return data[self.targetAttribute].unique().shape[0] == 1

	def mostLabel(self, data):
		return data[self.targetAttribute].value_counts().idxmax()

	def getChild(self, dataset, threshold, attributes, label):
		if (dataset.shape[0] > 0):
			child = self.buildDecisionTree(dataset, threshold, attributes)
		else:
			if self.binary:
				child = Tree.BinarySplitTree()
			else:
				child = Tree.MultiSplitTree()
			child.isLeaf = True
			child.label = label
		return child

	def buildDecisionTree(self, data, threshold, attributes):
		if self.binary:
			root = Tree.BinarySplitTree()
		else:
			root = Tree.MultiSplitTree()
		mostLabel = self.mostLabel(data)
		if self.isPure(data) or len(attributes) == 0 or data.shape[0] < threshold:
			root.isLeaf = True
			root.label = mostLabel
			return root
		else:
			root.isLeaf = False
			root.entropy = self.entropy(data)
			root.mostLabel = mostLabel
			attribute, value = self.bestAttribute(data, root.entropy, attributes)
			root.attribute = attribute
			newAttributes = attributes[:]
			newAttributes.remove(attribute)
			if self.binary:
				root.value = value
				root.leftChild = self.getChild(data.loc[data[attribute] < value], threshold, newAttributes, mostLabel)
				root.rightChild = self.getChild(data.loc[data[attribute] >= value], threshold, newAttributes, mostLabel)
			else:
				for newValue in data[attribute].unique():
					dataset = data.loc[data[attribute] == newValue]
					child = self.buildDecisionTree(dataset, threshold, newAttributes)
					child.value = newValue
					root.children.append(child)
		return root

	def classify(self, decisionTree, dataset):
		if decisionTree.isLeaf:
			return decisionTree.label
		else:
			value = dataset[decisionTree.attribute]
			if self.binary:
				if value < decisionTree.value:
					return self.classify(decisionTree.leftChild, dataset)
				else:
					return self.classify(decisionTree.rightChild, dataset)
			else:
				children = list(filter(lambda x: x.value == value, decisionTree.children))
				if len(children) > 0:
					return self.classify(children[0], dataset)
				else:
					return decisionTree.mostLabel

	def predict(self, decisionTree, data, test = False):
		error = 0
		for index, row in data.iterrows():
			actual = row[self.targetAttribute]
			prediction = self.classify(decisionTree, row)
			if test:
				self.actuals.append(actual)
				self.predictions.append(prediction)
			if prediction != actual:
				error += 1
		return error

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
			trainAccuracies = []
			testAccuracies = []
			self.actuals = []
			self.predictions = []
			for i in range(10):
				trainData = pd.concat(sets[0:i] + sets[i+1:10])
				testData = sets[i]
				decisionTree = self.buildDecisionTree(trainData, threshold, self.attributes)
				trainError = self.predict(decisionTree, trainData)
				trainAccuracy = (trainData.shape[0] - trainError) * 100 / trainData.shape[0]
				trainAccuracies.append(trainAccuracy)
				testError = self.predict(decisionTree, testData, True)
				testAccuracy = (testData.shape[0] - testError) * 100 / testData.shape[0]
				testAccuracies.append(testAccuracy)
			trainAccuracy = np.mean(trainAccuracies)
			testAccuracy = np.mean(testAccuracies)
			testStd = np.std(testAccuracies)
			print("MIN: {}, TRAIN ACCURACY: {}, TEST ACCURACY: {}, STANDARD DEVIATION: {}".
				format(m, trainAccuracy, testAccuracy, testStd))
			ConfusionMatrix(self.actuals, self.predictions).print_stats()


