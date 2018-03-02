from models import GMM, KMeans
from collections import defaultdict, namedtuple
import dataset
import numpy as np
from scipy.stats import multivariate_normal
import pylab as pl
from em_test import plotMOG, params


gmm_variants = [("full", "random"), ("diag", "random"),
                ("full", "kmeans"), ("diag", "kmeans")]

def average_likelihood(data_set, variants=None, mixtures=5):
    models = {}
    data = dataset.read_data(data_set)
    n = len(data)
    if not variants:
        variants = gmm_variants    
    ModelData = namedtuple('ModelData', ['avg_likelihood', 'model_obj'])
    for variant, init in variants:
        for i in xrange(1, mixtures + 1):
            gmm = fit_retry(i, data, variant, init)
            models[(variant, init, i)] = ModelData(
                avg_likelihood=gmm.likelihood / n, model_obj=gmm)
    return models


def rank_models():
    """ Rank models based on average log likelihood """
    data_sets = ["data_1_small", "data_2_small", "data_3_small"]
    for data_set in data_sets:
        avg_performance = defaultdict(float)
        models, test_results = dict(), {}
        for config, model_data in average_likelihood(data_set).iteritems():
            avg_performance[config] += model_data.avg_likelihood
            models[config] = model_data.model_obj
        ranked = sorted(avg_performance.iteritems(),
                        key=lambda x: x[1], reverse=True)
        large_data = dataset.read_data(data_set[0:-5]+"large")
        print "best model is {}".format(ranked[0])
        for config, _ in ranked:
            test_results[config] = test(models[config], large_data)
        test_ranked = sorted(test_results.iteritems(),
                             key=lambda x: x[1], reverse=True)
        best_model = models[ranked[0][0]]
        # plotMOG(large_data, params((best_model.pi, best_model.mu, best_model.sigma)))
        # pl.show()
        plot_rankings(avg_performance, test_results, title=data_set[0:-5])


def plot_rankings(train_results, test_results, title=""):
    """ Given train and test results for the GMM, plots their
        comparative results 
    """
    x, y = [], []
    label_idx, label_values = [], []
    for i, key in enumerate(train_results):
        label_idx.append(i)
        label_values.append(key)

    for config, ll in train_results.iteritems():
        x.append(label_values.index(config))
        y.append(-ll)
    pl.bar(x, y, width=0.5, color='b', align="center", label="train")
    x, y = [], []
    for config, ll in test_results.iteritems():
        x.append(label_values.index(config))
        y.append(-ll)
    pl.bar(x, y, width=0.5, color='r', label="test")
    pl.xticks(label_idx, map(str, label_values), rotation="vertical")
    pl.legend()
    pl.title("Train vs Test for {}".format(title))
    pl.tight_layout()
    pl.show()


def test(model, data):
    """ This functions uses the trained model and runs it on
        test data `data`
    """
    n, _ = data.shape
    likelihood = np.zeros((model.k, n))
    for j in xrange(model.k):
        likelihood[j, :] = model.pi[j] * \
            multivariate_normal(model.mu[j], model.sigma[j]).pdf(data)
    return np.sum(np.log(likelihood.sum(axis=0))) / n


def fit_retry(k, data, variant, init, attempt=1):
    if attempt >= 100:
        print("Failed!")
        g = GMM(k)
        g.likelihood = 0.0
        return g
    try:
        if init == 'kmeans':
            kmeans = KMeans(k)
            centroids, labels = kmeans.fit(data)
            gmm = GMM(k, mu=centroids, variant=variant,
                      progress=False, threshold=1e-3)
        else:
            gmm = GMM(k, variant=variant, progress=False, threshold=1e-3)
        gmm.fit(data)
        return gmm
    except ValueError as e:
        return fit_retry(k, data, variant, init, attempt=attempt + 1)


def leave_one_out_validation(data_set):
    data = dataset.read_data(data_set)
    cross_validation(data_set, data.shape[0] - 1)


def cross_validation(data_set, k):
    data = dataset.read_data(data_set)
    n, _ = data.shape
    if k == n - 1:
        fold = 1
    else:
        fold = n / k
    variants = [("full", "kmeans"), ("diag", "kmeans")]
    train_results, test_results, models = defaultdict(
        float), defaultdict(float), defaultdict(list)
    for i in xrange(0, n, fold):
        print("Running for {}".format(i))
        test_likelihoods = 0.0
        test_example = data[i:i+fold, :]
        train_examples = np.concatenate((data[:i, :], data[i + k + 1:, :]), axis=0)
        for config, model_data in average_likelihood(data_set).iteritems():
            t = test(model_data.model_obj, data)
            train_results[config] += t
            models[config].append((t, model_data.model_obj))
    ranked = sorted(train_results.iteritems(),
                    key=lambda x: x[1], reverse=True)
    print "Best model is {}".format(ranked[0])
    best_model = sorted(models[ranked[0][0]],
                        key=lambda x: x[0], reverse=True)[0][1]
    large_data = dataset.read_data(data_set[0:-5]+"large")
    print test(best_model, large_data)
    plotMOG(large_data, params((best_model.pi, best_model.mu, best_model.sigma)))
    pl.show()


if __name__ == '__main__':
    # rank_models()
    leave_one_out_validation("data_1_small")
    # leave_one_out_validation("data_2_small")
    # leave_one_out_validation("data_3_small")
    cross_validation("data_1_small", 4)
    # cross_validation("data_2_small", 4)
    # cross_validation("data_3_small", 6)