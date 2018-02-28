from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
import dataset
import sys

# Gaussian mixture model using Expectation-Maximization algorithm
class GMM(object):
    
    def __init__(self, k, d=2, pi=None, mu=None, sigma=None, iterations=1000):
        self.k = k # Number of gaussians in the mixture
        self.d = d # Number of variables
        self.pi = pi or self.init_pi()
        self.mu = mu or self.init_mu()
        self.sigma = sigma or self.init_sigma()
        self.iterations = iterations
        self.responsibility = None

    def init_pi(self):
        """ Initialize value for gaussian priors """
        return np.random.random(self.k)
    
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
        self.responsibility = self.responsibility / self.responsibility.sum(axis=0)  

    def maximization(self, X):
        n = len(X)

        # Maximize w.r.t. pi, means (mu), covariance
        for j in xrange(self.k):
            r_k = np.sum(self.responsibility[j, :])
            self.pi[j] = r_k / n
            self.mu[j] = np.dot(self.responsibility[j, :], X) / r_k 
            for i in range(n):
                diff = (X[i] - self.mu[j]).reshape(self.d, 1)
                self.sigma[j] += self.responsibility[j, i] * np.dot(diff, diff.T)
            self.sigma[j] /= r_k

    def fit(self, X):
        i = 0
        while i < self.iterations:
            self.expectation(X)
            self.maximization(X)
            i += 1
            self.print_progress(i)
        return self.pi, self.mu, self.sigma
    
    def print_progress(self, i):
        sys.stdout.write("Iterations {} \r".format(i))
        sys.stdout.flush()


if __name__ == '__main__':
    gmm = GMM(2)
    data = dataset.read_data()
    print(gmm.pi)
    print(gmm.mu)
    print(gmm.sigma)
    print(data.shape)
    gmm.fit(data)
    print("After")
    print(gmm.pi)
    print(gmm.mu)
    print(gmm.sigma)
    print(data.shape)
