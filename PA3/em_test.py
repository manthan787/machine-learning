import random
from numpy import *
from plotGauss2D import *
from models import GMM, KMeans, plotKMeans
import dataset

#############################
# Mixture Of Gaussians
#############################


class MOG:
    """ A simple class for a Mixture of Gaussians """

    def __init__(self, pi=0, mu=0, var=0):
        self.pi = pi
        self.mu = mu
        self.var = var

    def plot(self, color='black'):
        return plotGauss2D(self.mu, self.var, color=color)

    def __str__(self):
        return "[pi=%.2f,mu=%s, var=%s]" % (self.pi, self.mu.tolist(), self.var.tolist())
    __repr__ = __str__


colors = ('blue', 'yellow', 'black', 'red', 'cyan')


def plotMOG(X, param, colors=colors, title=""):
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
    pl.title("{} k={}".format(title, len(param)))
    pl.draw()


def plotData(X):
    pl.plot(X[:, 0:1].T[0], X[:, 1:2].T[0], 'gs')


def varMat(s1, s2, s12=0):
    return pl.array([[s1, s12], [s12, s2]])


def randomParams(X, m=2):
    # m is the number of mixtures
    # this function is used to generate random mixture, in your homework you should use EM algorithm to get real mixtures.
    (n, d) = X.shape
    # A random mixture...
    return [MOG(pi=1./m,
                mu=X[random.randint(0, n-1), :],
                var=varMat(3*random.random(), 3*random.random(), 3*random.random()-1.5))
            for i in range(m)]


def params(res, variant="full"):
    pi, mu, sigma = res
    dists = []
    for p, m, s in zip(pi, mu, sigma):
        dists.append(MOG(pi=p, mu=m, var=diag(s) if variant == "diag" else s))
    print dists
    return dists


def plot_loglikelihood(k, ll, title="", label="full"):
    """ Plot GMM log likelihood for different config """
    pl.plot(k, ll, label=label)
    pl.legend()
    pl.title(title)


def gmm_with_kmeans(k, data):
    data = dataset.read_data(data)
    kmeans = KMeans(k)
    centroids, labels = kmeans.fit(data)
    plotKMeans(data, kmeans.mu, labels)
    print centroids.shape
    gmm = GMM(k, mu=centroids)
    plotMOG(data, params(gmm.fit(data)), title="GMM with KMeans")

    gmm = GMM(k)
    plotMOG(data, params(gmm.fit(data)), title="GMM general")
    pl.show()


def gmm_test(k, variant, data_set, title):
    gmm = GMM(k, variant=variant)
    data = dataset.read_data(name=data_set)
    plotMOG(data, params(gmm.fit(data), variant), title=title)


def gmm_log_plot(d, type="full"):
    ks, ll = [], []
    for i in xrange(1, 5):
        data = dataset.read_data(d)
        if type == "kmeans":
            gmm = GMM(i, mu=KMeans(i).fit(data)[0])
        elif type == "diag":
            gmm = GMM(i, variant="diag")
        else:
            gmm = GMM(i)
        gmm.fit(data)
        ks.append(i)
        ll.append(-gmm.likelihood)
        print("Likelihood for k = {} => {}".format(i, gmm.likelihood))
    plot_loglikelihood(ks, ll, label=type)
    pl.ylabel("Log Likelihood")
    pl.xlabel("Number of mixtures")
    pl.draw()


if __name__ == '__main__':
    # gmm_with_kmeans(3, "data_2_large")

    # Q1.3
    # gmm_log_plot("data_1_large", type="data_1_large")
    # gmm_log_plot("data_2_large", type="data_2_large")
    # gmm_log_plot("data_3_large", type = "data_3_large")
    gmm_log_plot("data_1_small", type="data_1_small")
    gmm_log_plot("data_2_small", type="data_2_small")
    gmm_log_plot("data_3_small", type="data_3_small")
    # gmm_log_plot("data_3_large")
    # gmm_log_plot("data_3_large", type="kmeans")
    # gmm_log_plot("data_3_large", type="diag")
    pl.show()
    # gmm_test(5, "diag", "data_3_large", "GMM with full covariance")
    # gmm_test(5, "full", "data_3_large", "GMM with diagonal covariance")
    # pl.show()
