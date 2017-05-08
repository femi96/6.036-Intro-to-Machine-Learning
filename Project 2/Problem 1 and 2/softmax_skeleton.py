import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
    # x*theta
    t = np.dot(theta, np.transpose(X))
    # kxn array, k labels, n points
    h = np.zeros(t.shape)
    # for each point
    for i in range(0, t.shape[1]):
        # overall time element
        tn = t[:,i]/tempParameter
        # avoid overflow error with c constant
        c = tn.max()
        e = np.exp(tn-c)
        # change h column for sample n
        h[:,i] = e/np.sum(e)
    return h

def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
    # probability, kxn array, k labels, n points
    h = computeProbabilities(X, theta, tempParameter)
    l = np.log(h)
    # cost starts at 0
    cost = 0
    # for each point i, 0 to n
    n = Y.size
    k = np.max(Y)+1;
    for i in range(0, n):
        # for each label j, 0 to k
        for j in range(0, k):
            if Y[i] == j:
                cost -= l[j,i]/n
    cost += lambdaFactor/2*np.sum(np.power(theta, 2))
    return cost

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):
    # k labels, n points, d features
    # probability, kxn
    p = computeProbabilities(X, theta, tempParameter)
    # new theta kxd
    theta_new = np.copy(theta)
    # for each label j, in range 0 to k
    for j in range(0, theta.shape[0]):
        # gradient theta_j, dx1
        grad = lambdaFactor*theta[j,:]
        # for each point i, in range 0 to n
        n = Y.size
        for i in range(0, n):
            # xi * ([y == p] - p)/(nT), dx1
            if Y[i] == j:
                grad -= X[i,:]*(1-p[j,i])/(n*tempParameter)
            else:
                grad -= X[i,:]*(-p[j,i])/(n*tempParameter)
        theta_new[j,:] -= grad*alpha
    return theta_new

def updateY(trainY, testY):
    return trainY % 3, testY % 3

def computeTestErrorMod3(X, Y, theta, tempParameter):
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels % 3 == Y)

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        print(i)
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
    return theta, costFunctionProgression
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
