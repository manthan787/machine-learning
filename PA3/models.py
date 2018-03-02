from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
import dataset
import sys
from plotGauss2D import *
from scipy.special import logsumexp


class GMM(object):
    """ Gaussian mixture model using Expectation-Maximization algorithm """

    def __init__(self, k, d=2, pi=None, mu=None, sigma=None, iterations=30,
                 threshold=1e-4, variant="full", progress=True):
        self.k = k  # Number of gaussians in the mixture
        self.d = d  # Number of variables
        self.variant = variant
        self.pi = pi if pi is not None else self.init_pi()
        self.mu = mu if mu is not None else self.init_mu()
        self.sigma = sigma if sigma is not None else self.init_sigma()
        self.iterations = iterations
        self.responsibility = None
        self.likelihood = 1.0
        self.threshold = threshold
        self.converged = False
        self.progress = progress

    def init_pi(self):
        """ Initialize value for gaussian priors """
        return np.ones((self.k, 1)) * (1 / self.k)

    def init_mu(self):
        """ Initialize value for gaussian means """
        return np.random.random((self.k, self.d))

    def init_sigma(self):
        """ Initialize value for covariance matrices """
        if self.variant == 'diag':
            return np.ones((self.k, self.d))
        return np.repeat([np.eye(self.d)], self.k, axis=0)

    def expectation(self, X):
        if self.responsibility is None:
            n = len(X)
            self.responsibility = np.zeros((self.k, n))

        for j in xrange(self.k):
            self.responsibility[j, :] = self.pi[j] * multivariate_normal(
                self.mu[j], self.sigma[j]).pdf(X)
        self.responsibility = self.responsibility / \
            self.responsibility.sum(axis=0)

    def maximization(self, X):
        n, d = X.shape

        # Maximize w.r.t. pi, means (mu), covariance
        for j in xrange(self.k):
            r_k = np.sum(self.responsibility[j, :])
            self.pi[j] = r_k / n
            self.mu[j] = np.dot(self.responsibility[j, :], X) / r_k
            for i in range(n):
                if self.variant == 'full':
                    diff = (X[i] - self.mu[j]).reshape(self.d, 1)
                    self.sigma[j] += self.responsibility[j, i] * \
                        np.dot(diff, diff.T)
                
                if self.variant == 'diag':
                    self.sigma[j] += self.responsibility[j, i] * \
                                ((X[i] - self.mu[j]) ** 2)
            self.sigma[j] = self.sigma[j] / r_k
        self.calculate_likelihood(X, n)

    def calculate_likelihood(self, X, n):
        likelihood = np.zeros((self.k, n))
        for j in xrange(self.k):
            likelihood[j, :] = self.pi[j] * \
                multivariate_normal(self.mu[j], self.sigma[j]).pdf(X)
        likelihood = np.sum(np.log(likelihood.sum(axis=0)))
        delta = (likelihood - self.likelihood) / self.likelihood
        if abs(delta) < self.threshold:
            self.converged = True
        self.likelihood = likelihood

    def fit(self, X):
        i = 0
        while i < self.iterations and not self.converged:
            self.expectation(X)
            self.maximization(X)
            i += 1
            if self.progress:
                self.print_progress(i, self.likelihood)
        return self.pi, self.mu, self.sigma

    def print_progress(self, i, l):
        sys.stdout.write("Iterations {} Likelihood: {} \r".format(i, l))
        sys.stdout.flush()


def fit_retry(k, data, attempt=1):
    if attempt >= 100:
        print("Failed!")
        return
    try:
        gmm = GMM(k, iterations=200)
        gmm.fit(data)
        return gmm
    except ValueError as e:
        fit_retry(k, data, attempt=attempt + 1)


class KMeans(object):

    def __init__(self, k, max_iter=1000, progress=False):
        self.k = k  # Number of clusters
        self.mu = None  # Centroids of clusters
        self.max_iter = max_iter
        self.progress = progress

    def fit(self, X):
        n, d = X.shape
        labels = np.zeros(n)
        self.mu, new_mu = np.random.random((self.k, d)), np.zeros((self.k, d))
        iters = 0
        while iters < self.max_iter:
            # Assign labels based on distances to centroids
            for i in xrange(n):
                labels[i] = np.argmin(np.linalg.norm(self.mu - X[i], axis=1))

            # Update centroids
            for i in xrange(self.k):
                cluster_i = np.argwhere(labels == i)
                new_mu[i, :] = np.take(X, cluster_i, axis=0).mean(axis=0)

            if np.all(self.mu == new_mu):
                # print "converged", iters
                break

            self.mu = new_mu
            new_mu = np.zeros((self.k, d))
            iters += 1
            if self.progress:
                self.print_progress(iters)
        return self.mu, labels

    def print_progress(self, i):
        sys.stdout.write("Iterations {} \r".format(i))
        sys.stdout.flush()


def gmm_test():
    for i in xrange(1, 11):
        gmm = GMM(i, iterations=50)
        data = dataset.read_data("data_2_large")
        gmm.fit(data)
        print("Likelihood for k = {} => {}".format(i, gmm.likelihood))


def plotKMeans(X, centroids, labels):
    pl.scatter(X[:, 0:1].T[0], X[:, 1:2].T[0], c=labels)
    pl.plot(centroids[:, 0:1].T[0], centroids[:, 1:2].T[0], "ro")
    pl.draw()


def kmeans_test():
    kmeans = KMeans(3)
    data = dataset.read_data("data_2_large")
    _, labels = kmeans.fit(data)
    plotKMeans(data, kmeans.mu, labels)


def gmm_with_kmeans(k, data):
    kmeans = KMeans(k)
    data = dataset.read_data(data)
    centroids, labels = kmeans.fit(data)
    plotKMeans(data, kmeans.mu, labels)
    # gmm = GMM(k)
    # gmm.fit()

if __name__ == '__main__':
    gmm_with_kmeans(3) 
    gmm.fit("mystery_1")
    # gmm = GMM(2, variant="diag")
    # gmm.fit(dataset.read_data("data_1_large"))
