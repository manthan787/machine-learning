from numpy import *
from plotBoundary import *
import cvxopt
import numpy as np

# parameters
DATA_PATH = "data/"
def get_data(f):
    data = np.loadtxt(open(DATA_PATH + f, "rb"), delimiter=" ", skiprows=0)
    return data[:, :-1], data[:, -1].reshape(len(data), 1)

def get_train_data(sep="ls"):
    f = "data_{}_train.csv".format(sep)
    return get_data(f)

def get_validation_data(sep="ls"):
    f = "data_{}_validate.csv".format(sep)
    return get_data(f)

def kernel_matrix(X, kernel):
    ''' Given features `X` and `kernel` create the kernel matrix '''
    return kernel(X, X)

def linear_kernel(a, b):
    return np.dot(a, b.T)

def polynomial_kernel(a, b):
    return np.dot(a, b) ** 2

def gaussian_kernel(a, b, sigma=0.05):
    return np.exp(-np.sqrt((np.linalg.norm(a - b) ** 2) / (2 * sigma ** 2)))

def svm_train_linear(X, Y, C=1, kernel=linear_kernel, data="ls"):
    ''' Train SVM classifier using the dual form of optimization '''
    kernel_mat = kernel_matrix(X, linear_kernel)
    n, d = X.shape

    P = cvxopt.matrix(np.dot(Y, Y.T) * kernel_mat)
    q = cvxopt.matrix(-np.ones(n))
    G = cvxopt.matrix(np.vstack((np.diag(-np.ones(n)), np.diag(np.ones(n)))))
    h = cvxopt.matrix(np.vstack((np.zeros(n).reshape(n, 1), (C * np.ones(n)).reshape(n, 1))))
    A = cvxopt.matrix(Y, (1, n))
    b = cvxopt.matrix(0.0)

    alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
    support_vectors_idx = alphas > 0
    a_k = alphas[support_vectors_idx]
    x_k = X[support_vectors_idx]
    y_k = Y[support_vectors_idx]
    weights = np.sum(alphas.reshape(n, 1) * Y * X, axis=0)
    bias = np.mean(y_k - np.dot(x_k, weights))
    print bias
    return a_k, x_k, y_k, weights, bias


def svm_train(X, Y, C=1, kernel=linear_kernel, data="ls"):
    ''' Train SVM classifier using the dual form of optimization '''
    kernel_mat = kernel_matrix(X, linear_kernel)
    n, d = X.shape

    P = cvxopt.matrix(np.dot(Y, Y.T) * kernel_mat)
    q = cvxopt.matrix(-np.ones(n))
    G = cvxopt.matrix(np.vstack((np.diag(-np.ones(n)), np.diag(np.ones(n)))))
    h = cvxopt.matrix(np.vstack((np.zeros(n).reshape(n, 1), (C * np.ones(n)).reshape(n, 1))))
    A = cvxopt.matrix(Y, (1, n))
    b = cvxopt.matrix(0.0)

    alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
    support_vectors_idx = alphas > 0
    a_k = alphas[support_vectors_idx]
    x_k = X[support_vectors_idx]
    y_k = Y[support_vectors_idx]
    biases = []
    for y, x in zip(y_k, x_k):
        bias = 0
        for a_i, x_i, y_i in zip(a_k, x_k, y_k):
            bias += a_i * y_i * kernel(x_i, x)
        biases.append(y - np.sign(bias))
    bias = np.mean(biases)
    print bias
    return a_k, x_k, y_k, [0], bias

def predictSVM(x):
    global weights, bias
    return np.dot(weights.T, x) + bias

def mistakes(x, y):
    global weights, bias
    pred = np.dot(x, weights) + bias
    pred = np.where(pred > 0, 1, -1).reshape(len(x), 1)
    return np.sum( pred != y )

def predictSVMPoly(x):
    global a_k, x_k, y_k, bias
    ans = bias
    for a_i, x_i, y_i in zip(a_k, x_k, y_k):
        ans += a_i * y_i * polynomial_kernel(x_i, x)
    # return np.sign(ans).item()
    return ans

def predictSVMGaussian(x):
    global a_k, x_k, y_k, bias
    ans = bias
    for a_i, x_i, y_i in zip(a_k, x_k, y_k):
        ans += a_i * y_i * gaussian_kernel(x_i, x)
    return ans

def svm_linear(data="nonlin"):
    print "Linear SVM ..."
    global a_k, x_k, y_k, bias, weights
    X, Y = get_train_data(sep=data)
    validate_X, validate_Y = get_validation_data(sep=data)
    a_k, x_k, y_k, weights, bias = svm_train_linear(X, Y, C=0.05, kernel=linear_kernel)
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train Linear')
    plotDecisionBoundary(validate_X, validate_Y, predictSVM, [-1, 0, 1], title = 'SVM Validate Linear')

def svm_poly(data="nonlin"):
    print "Polynomial ..."
    global a_k, x_k, y_k, bias
    X, Y = get_train_data(sep=data)
    validate_X, validate_Y = get_validation_data(sep=data)
    a_k, x_k, y_k, weights, bias = svm_train(X, Y, C=0.5, kernel=polynomial_kernel)
    plotDecisionBoundary(X, Y, predictSVMPoly, [-1, 0, 1], title = 'SVM Train Poly')
    plotDecisionBoundary(validate_X, validate_Y, predictSVMPoly, [-1, 0, 1], title = 'SVM Validate Poly')

def svm_gaussian(data="nonlin"):
    print "Gaussian ..."
    global a_k, x_k, y_k, bias
    X, Y = get_train_data(sep=data)
    validate_X, validate_Y = get_validation_data(sep=data)
    a_k, x_k, y_k, weights, bias = svm_train(X, Y, C=0.5, kernel=gaussian_kernel)
    plotDecisionBoundary(X, Y, predictSVMGaussian, [-1, 0, 1], title = 'SVM Train Gaussian')
    plotDecisionBoundary(validate_X, validate_Y, predictSVMGaussian, [-1, 0, 1], title = 'SVM Validate Gaussian')

# Gaussian overfits at C = 0.7 sigma=0.05
# svm_linear("nls")
# svm_gaussian("nonlin")
# svm_gaussian("nls")
svm_poly("nls")

