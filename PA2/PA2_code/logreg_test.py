from __future__ import division
from numpy import *
from plotBoundary import *
import numpy as np
import sys

DATA_PATH = "data/"
weights = None

def get_data(f):
    data = np.loadtxt(open(DATA_PATH + f, "rb"), delimiter=" ", skiprows=0)
    ones = np.ones((len(data), 1))
    return np.concatenate((ones, data[:, :-1]), 1), data[:, -1].reshape(len(data), 1)

def get_train_data(sep="ls"):
    f = "data_{}_train.csv".format(sep)
    return get_data(f)

def get_validation_data(sep="ls"):
    f = "data_{}_validate.csv".format(sep)
    return get_data(f)

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    global weights
    x = np.concatenate((np.array([1]), x), 1)
    e = np.exp(np.dot(weights.T, x))
    return e / (e + 1)

def predictLRPoly(x):
    global weights
    x = np.concatenate((np.array([1]), x, \
                        np.array([x[0] ** 2]), np.array([x[1] ** 2]), \
                        np.array([x[0] * x[1]])), 1)
    e = np.exp(np.dot(weights.T, x))
    return e / (e + 1)

# Carry out training.
def gradient_descent(gradient, init, step_size=0.01, train_x=None, train_y=None, \
                     convergence_criteria=0.0008, loss=None, l = 0, \
                     predict = predictLR, real_time=False):
    ''' Gradient descent wrapper function
        `gradient` is a function that calculates gradient for the objective function
        `init` is the initial guess for parameters
        `step_size` is the learning rate
        `train_x` and `train_y` is train data if required
        `convergence_criteria` is the threshold for difference between two consecutive
                               iterations
        `loss` is loss function to print loss function after each step
        `l` is lambda for regularization
        `real_time` is a flag for whether the plot should be plotted in real time or not
    '''
    global weights
    i = 0
    params, previous = init, init # initialize params
    diff = np.array([10]* len(init))
    while not all(diff < convergence_criteria):
        grad, t = gradient(params, train_x, train_y, l) # calculate gradient
        previous = np.copy(params)
        params += step_size * grad
        # print params
        diff = abs(params - previous)
        i += 1
        if loss:
            print "Loss", loss(params, train_x, train_y)

        if real_time:
            if i % 1000 == 0:
                weights = params
                plotDecisionBoundary(train_x[:, 1:], train_y, predict, [0.5], title = 'LR Train',\
                 real_time=True)
        sys.stdout.write("Iterations {} \r".format(i))
        sys.stdout.flush()
    print "Total Iterations", i
    return params

def basis_expansion(x):
    D = len(x[0])
    return np.concatenate((x, np.square(x)[:, 1:], (x[:, 1] * x[:, 2]).reshape(len(x), 1)), 1)

def logistic_loss(params, train_x, train_y, l = 0):
    ''' Gradient of logistic loss function '''
    # e = np.exp(-1 * np.multiply(train_y, np.dot(train_x, params)))
    e = np.exp(np.multiply(train_y, np.dot(train_x, params)))
    t = - np.log(1 + e)
    return np.sum(np.multiply(np.exp(t), np.multiply(train_x, train_y)) \
                  + (2 * l * params.T), axis=0).reshape(len(train_x[0]), 1), e

def mistakes(params, x, y):
    ''' Count number of mistakes made by trained parameters
        on given data `x` and `y`.
    '''
    e = np.exp(np.dot(x, params))
    sigmoid_res = e / (e + 1)
    decision = np.where(sigmoid_res < 0.5, -1, 1)
    return np.sum( decision != y )

def run(data="ls"):
    global weights
    print "Training ..."
    X, Y = get_train_data(sep=data)
    validate_X, validate_Y = get_validation_data(sep=data)
    weights = gradient_descent(logistic_loss, np.random.rand(len(X[0]), 1), \
                            train_x=X, train_y=Y, convergence_criteria=0.00008, \
                            step_size = 0.01, l = 0, real_time=False)
    print weights
    print "Classification mistakes on train data:", mistakes(weights, X, Y)
    print "Classification mistakes on validation data:", mistakes(weights, validate_X, validate_Y)
    # Plot training results
    plotDecisionBoundary(X[:, 1:], Y, predictLR, [0.5], title = 'LR Train')
    plotDecisionBoundary(validate_X[:, 1:], validate_Y, predictLR, [0.5], title = 'LR Validation')

def run_expansion(data="ls"):
    global weights
    X, Y = get_train_data(sep=data)
    poly_x = basis_expansion(X)
    validate_X, validate_Y = get_validation_data(sep=data)
    validate_X = basis_expansion(validate_X)
    weights = gradient_descent(logistic_loss, np.random.rand(len(poly_x[0]), 1), \
                            train_x=poly_x, train_y=Y, convergence_criteria=0.00001, \
                            step_size = 0.0000001, l = 0.05, predict=predictLRPoly, real_time=True)
    print weights
    print "Classification mistakes on train data:", mistakes(weights, poly_x, Y)
    print "Classification mistakes on validation data:", mistakes(weights, validate_X, validate_Y)
    # Plot training results
    plotDecisionBoundary(poly_x[:, 1:], Y, predictLRPoly, [0.5], title = 'LR Train')
    plotDecisionBoundary(validate_X[:, 1:], validate_Y, predictLRPoly, [0.5], title = 'LR Validation')

if __name__ == '__main__':
    run_expansion("nonlin")
