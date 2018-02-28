import random
from numpy import *
from plotGauss2D import *
from EM import GMM
import dataset

#############################
# Mixture Of Gaussians
#############################

# A simple class for a Mixture of Gaussians
class MOG:
    def __init__(self, pi = 0, mu = 0, var = 0):
        self.pi = pi
        self.mu = mu
        self.var = var
    def plot(self, color = 'black'):
        return plotGauss2D(self.mu, self.var, color=color)
    def __str__(self):
        return "[pi=%.2f,mu=%s, var=%s]"%(self.pi, self.mu.tolist(), self.var.tolist())
    __repr__ = __str__
 
colors = ('blue', 'yellow', 'black', 'red', 'cyan')

def plotMOG(X, param, colors = colors):
    fig = pl.figure()                   # make a new figure/window
    ax = fig.add_subplot(111, aspect='equal')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
    for (g, c) in zip(param, colors[:len(param)]):
        e = g.plot(color=c)
        ax.add_artist(e)
    plotData(X)
    pl.show()

def plotData(X):
    pl.plot(X[:,0:1].T[0],X[:,1:2].T[0], 'gs')
 
def varMat(s1, s2, s12 = 0):
    return pl.array([[s1, s12], [s12, s2]])
 
def randomParams(X, m=2):
    # m is the number of mixtures
    # this function is used to generate random mixture, in your homework you should use EM algorithm to get real mixtures.
    (n, d) = X.shape
    # A random mixture...
    return [MOG(pi=1./m, 
                mu=X[random.randint(0,n-1),:], 
                var=varMat(3*random.random(), 3*random.random(), 3*random.random()-1.5)) \
            for i in range(m)]


def params(res):
    pi, mu, sigma = res
    dists = []
    for p, m, s in zip(pi, mu, sigma):
        dists.append(MOG(pi=p, mu=m, var=s))
    return dists


if __name__ == '__main__':
    gmm = GMM(3, iterations=30)
    dataset_name = "data_3_large"
    data = dataset.read_data(name=dataset_name)
    plotMOG(data, params(gmm.fit(data)))