from __future__ import division
from tabulate import tabulate
from copy import deepcopy
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Clustering:
	def __init__(self, dataset, k, GMM = False, tolerance=0.00001, ep=100):
		self.dataset = dataset
		self.k = k
		self.ep = ep
		self.tolerance = tolerance
		self.GMM = GMM


	def GuassianNormal(self, x, c, s):
		return np.exp(-0.5 * np.einsum('ij, ij -> i', x - c, np.dot(np.linalg.inv(s), (x - c).T).T)) / (((2 * np.pi)**(len(c) / 2)) * (np.linalg.det(s)**0.5))

	def SSE(self, x, cluster, c):
		sse = 0.0
		for i in range(len(x)):
			sse += np.sum(np.square(np.subtract(x[i], c[int(cluster[i])])))
		return sse

	def NMI(self, y, cluster, c):
		labelClasses = collections.Counter(y)
		clusterClasses = collections.Counter(cluster)
		hy = 0.0
		for label in labelClasses:
			labelPb = labelClasses[label] / len(y)
			if labelPb > 0.0:
				hy -= labelPb * np.log2(labelPb)

		cPbs = {}
		hc = 0.0
		for i in range(len(c)):
			cPb = clusterClasses[i] / len(cluster)
			cPbs[i] = cPb
			if cPb > 0.0:
				hc -= cPb * np.log2(cPb)

		yc = 0.0
		for i in range(len(c)):
			_label = [y[j] for j in range(len(y)) if cluster[j] == i]
			_labelClasses = collections.Counter(_label)
			cy = 0.0
			for label in _labelClasses:
				labelPb = _labelClasses[label] / len(_label)
				if labelPb > 0.0:
					cy += labelPb * np.log2(labelPb)
			yc -= cPbs[i] * cy

		return (hy - yc) * 2 / (hy + hc)


	def validate(self):
		SSEs = []
		NMIs = []

		x = self.dataset.iloc[:, :-1].values
		y = self.dataset.iloc[:, -1].values

		header = ["k", "SSE", "NMI"]
		table = []

		for k in self.k:
			c = x[np.random.choice(len(x), k, False), :]
			cluster = np.zeros(len(x))
			err = self.tolerance + 1

			if self.GMM:
				s = [np.eye(x.shape[1])] * k
				w = [1 / k] * k
				r = np.zeros((len(x), k))
				likelihood = 0


				for ep in range(self.ep):
					if err > self.tolerance:
						for _k in range(k):
							r[:, _k] = w[_k] * self.GuassianNormal(x, c[_k], s[_k])
						r = (r.T / np.sum(r, axis=1)).T
						p = np.sum(r, axis=0)
						for _k in range(k):
							c[_k] = 1 / p[_k] * np.sum(r[:, _k] * x.T, axis=1).T
							s[_k] = np.array(1 / p[_k] * np.dot(np.multiply(np.matrix(x - c[_k]).T, r[:, _k]), np.matrix(x - c[_k])))
							w[_k] = 1 / len(x) * p[_k]

						_likelihood = np.sum(np.log(np.sum(r, axis=1)))
						err = _likelihood - likelihood
						likelihood = _likelihood

					else:
						break

				for i in range(len(x)):
					cluster[i] = np.argmax(r[i])





			else:
				_c = np.zeros(c.shape)
			
				for ep in range(self.ep):
					if err > self.tolerance:
						for i in range(len(x)):
							cluster[i] = np.argmin(np.linalg.norm(x[i] - c, axis=1))
						_c = deepcopy(c)

						for i in range(k):
							p = [x[j] for j in range(len(x)) if cluster[j] == i]
							if len(p) > 0:
								c[i] = np.mean(p, axis=0)

						err = np.argmin(np.linalg.norm(c - _c))

					else:
						break

			sse = self.SSE(x, cluster, c)
			nmi = self.NMI(y, cluster, c)
			SSEs.append(sse)
			NMIs.append(nmi)

			table.append([k, sse, nmi])

		print(tabulate(table, header, tablefmt='grid'))

		plt.figure()
		plt.plot(self.k, SSEs, 'bo-')
		plt.ylabel('SSE')
		plt.xlabel('k')

		if self.GMM:
			plt.title("GMM Clustering SSE")
			plt.show()

			plt.figure()
			plt.plot(self.k, NMIs, 'ro-')
			plt.ylabel('NMI')
			plt.xlabel('k')
			plt.title("GMM Clustering NMI")

		else:
			plt.title("KMeans Clustering SSE")

		plt.show()

