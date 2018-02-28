from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
import dataset
import sys


class GMM(object):
    """ Gaussian mixture model using Expectation-Maximization algorithm """

    def __init__(self, k, d=2, pi=None, mu=None, sigma=None, iterations=100, threshold=1e-3):
        self.k = k  # Number of gaussians in the mixture
        self.d = d  # Number of variables
        self.pi = pi or self.init_pi()
        self.mu = mu or self.init_mu()
        self.sigma = sigma or self.init_sigma()
        self.iterations = iterations
        self.responsibility = None
        self.likelihood = 1.0
        self.threshold = threshold
        self.converged = False

    def init_pi(self):
        """ Initialize value for gaussian priors """
        return np.ones((self.k, 1)) * (1 / self.k)

    def init_mu(self):
        """ Initialize value for gaussian means """
        return np.random.random((self.k, self.d))

    def init_sigma(self):
        """ Initialize value for covariance matrices """
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
        n = len(X)

        # Maximize w.r.t. pi, means (mu), covariance
        for j in xrange(self.k):
            r_k = np.sum(self.responsibility[j, :])
            self.pi[j] = r_k / n
            self.mu[j] = np.dot(self.responsibility[j, :], X) / r_k
            for i in range(n):
                diff = (X[i] - self.mu[j]).reshape(self.d, 1)
                self.sigma[j] += self.responsibility[j, i] * \
                    np.dot(diff, diff.T)
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


if __name__ == '__main__':
    for i in xrange(1, 11):
        gmm = GMM(i, iterations=50)
        data = dataset.read_data("data_2_large")
        gmm.fit(data)
        print("Likelihood for k = {} => {}".format(i, gmm.likelihood))
