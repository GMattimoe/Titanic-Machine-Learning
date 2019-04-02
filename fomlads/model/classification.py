import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from fomlads.data.function import logistic_sigmoid
from fomlads.model.density_estimation import max_lik_mv_gaussian

def project_data(data, weights):
    """
    Projects data onto single dimension according to some weight vector

    parameters
    ----------
    data - a 2d data matrix (shape NxD array-like)
    weights -- a 1d weight vector (shape D array like)

    returns
    -------
    projected_data -- 1d vector (shape N np.array)
    """
    N, D = data.shape
    data = np.matrix(data)
    weights = np.matrix(weights).reshape((D,1))
    projected_data = np.array(data*weights).flatten()
    return projected_data

def fisher_linear_discriminant_projection(inputs, targets):
    """
    Finds the direction of best projection based on Fisher's linear discriminant

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector corresponding to Fisher's linear 
        discriminant
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    # find maximum likelihood approximations to the two data-sets
    m0, S_0 = max_lik_mv_gaussian(inputs0)
    m1, S_1 = max_lik_mv_gaussian(inputs1)
    # convert the mean vectors to column vectors (type matrix)
    m0 = np.matrix(m0).reshape((D,1))
    m1 = np.matrix(m1).reshape((D,1))
    # calculate the total within-class covariance matrix (type matrix)
    S_W = np.matrix(S_0 + S_1)
    # calculate weights vector
    weights = np.array(np.linalg.inv(S_W)*(m1-m0))
    # normalise
    weights = weights/np.sum(weights)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, weights))
    projected_m1 = np.mean(project_data(inputs1, weights))
    if projected_m0 > projected_m1:
        weights = -weights
    return weights

def robust_fisher_linear_discriminant_projection(inputs, targets, lambda_val):
    """
    Finds the direction of best projection based on Fisher's linear discriminant

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector corresponding to Fisher's linear 
        discriminant
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    # find maximum likelihood approximations to the two data-sets
    m0, S_0 = max_lik_mv_gaussian(inputs0)
    m1, S_1 = max_lik_mv_gaussian(inputs1)
    # convert the mean vectors to column vectors (type matrix)
    m0 = np.matrix(m0).reshape((D,1))
    m1 = np.matrix(m1).reshape((D,1))
    # calculate the total within-class covariance matrix (type matrix)
    S_W = np.matrix(S_0 + S_1)
    dim1,dim2 = S_W.shape
    S_W += lambda_val*np.identity(dim1)
    # calculate weights vector
    weights = np.array(np.linalg.inv(S_W)*(m1-m0))
    # normalise
    weights = weights/np.sum(weights)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, weights))
    projected_m1 = np.mean(project_data(inputs1, weights))
    if projected_m0 > projected_m1:
        weights = -weights
    return weights

def maximum_separation_projection(inputs, targets):
    """
    Finds the projection vector that maximises the distance between the 
    projected means

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    # find maximum likelihood approximations to the two data-sets
    m0,_ = max_lik_mv_gaussian(inputs0)
    m1,_ = max_lik_mv_gaussian(inputs1)
    # calculate weights vector
    weights = m1-m0
    return weights



def shared_covariance_model_fit(inputs, targets):
    """
    Finds the maximum-likelihood parameters of the shared covariance model for
    two classes

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    pi - the prior probability for class 1
    mean0 - the mean of class 0's data
    mean1 - the mean of class 1's data
    covmtx - the shared covariance matrix 
    """
    if len(inputs.shape) == 1:
        inputs = inputs.rehape(inputs.size,1)
    N, D = inputs.shape
    inputs0 = inputs[targets==0,:]
    inputs1 = inputs[targets==1,:]
    N0 = inputs0.shape[0]
    N1 = inputs1.shape[0]
    pi = N1/N
    mean0, S0 = max_lik_mv_gaussian(inputs0)
    mean1, S1 = max_lik_mv_gaussian(inputs1)
    covmtx = (N0/N)*S0 + (N1/N)*S1
    return pi, mean0, mean1, covmtx


def shared_covariance_model_predict(inputs, pi, mean0, mean1, covmtx):
    """
    Predicts a class label for a collection of datapoints based on the shared
    covariance generative model.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    pi - the prior probability for class 1
    mean0 - the mean of class 0's data
    mean1 - the mean of class 1's data
    covmtx - the shared covariance matrix 

    returns
    -------
    outputs - a 1d array of predictions, one per datapoint.
        prediction labels are 1 for class 1, and 0 for class 0
    """
    # calculate the class densities p(xn|C0) and p(xn|C1) for every data-point
    class0_densities = stats.multivariate_normal.pdf(inputs, mean0, covmtx)    
    class1_densities = stats.multivariate_normal.pdf(inputs, mean1, covmtx)
    # now evaluate the posterior class probability p(C1|xn) for every data-point
    posterior_probs = \
        (pi*class1_densities)/(pi*class1_densities+(1-pi)*class0_densities)
    return (posterior_probs >= 0.5).astype(int)

def logistic_regression_fit(
        inputs, targets, weights0=None, threshold=1e-8):
    """
    Fits a set of weights to the logistic regression model using the iteratively
    reweighted least squares (IRLS) method (Rubin, 1983)

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a set of weights for the model
    """
    # reshape the matrix for 1d inputs
    if len(inputs.shape) == 1:
        inputs = inputs.reshape((inputs.size,1))
    N, D = inputs.shape
    inputs = np.matrix(inputs)
    targets = np.matrix(targets.reshape((N,1)))
    # initialise the weights
    if weights0 is None:
        weights = np.matrix(
            np.random.multivariate_normal(np.zeros(D), 0.0001*np.identity(D)))
    else:
        weights = np.matrix(weights0)
    weights = weights.reshape((D,1))
    # initially the update magnitude is set as larger than the threshold
    update_magnitude = 2*threshold
    while update_magnitude > threshold:
        # calculate the current prediction vector for weights
        predicts = logistic_regression_prediction_probs(inputs, weights)
        # the diagonal reweighting matrix (easier with predicts as flat array)
        R = np.matrix(np.diag(predicts*(1-predicts)))
        # reshape predicts to be same form as targets
        predicts = np.matrix(predicts).reshape((N,1))
        # Calculate the Hessian inverse
        mat = inputs.T*R*inputs
        H_inv = np.linalg.inv(inputs.T*R*inputs)
        # update the weights
        new_weights = weights - H_inv*inputs.T*np.matrix(predicts-targets)
        # calculate the update_magnitude
        update_magnitude = np.sqrt(np.sum(np.array(new_weights-weights)**2))
        # update the weights
        weights = new_weights
    return weights

def logistic_regression_predict(inputs, weights):
    """
    Get deterministic class prediction vector from the logistic regression model.

    parameters
    ----------
    inputs - input data (or design matrix) as 2d array
    weights - a set of model weights
    """
    prediction_probs = logistic_regression_prediction_probs(inputs, weights)
    return (prediction_probs > 0.5).astype(int)

def logistic_regression_prediction_probs(inputs, weights):
    """
    Get prediction probability vector from the logistic regression model.

    parameters
    ----------
    inputs - input data (or design matrix) as 2d array
    weights - a set of model weights
    """
    N, D = inputs.shape
    weights = np.matrix(weights).reshape((D,1))
    inputs = np.matrix(inputs)
    return logistic_sigmoid(np.array(inputs*weights).flatten())

