import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def ml_weights(inputs, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputs)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def regularised_ml_weights(
        inputs, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (reg_param)
    """
    Phi = np.matrix(inputs)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(reg_param*I + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()


def linear_model_predict(inputs, weights):
    ys = np.matrix(inputs)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

def calculate_weights_posterior(inputs, targets, beta, m0, S0):
    """
    Calculates the posterior distribution (multivariate gaussian) for weights
    in a linear model.

    parameters
    ----------
    inputs - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    targets - 1d (N)-array of target values
    beta - the known noise precision
    m0 - prior mean (vector) 1d-array (or array-like) of length K
    S0 - the prior covariance matrix 2d-array

    returns
    -------
    mN - the posterior mean (vector)
    SN - the posterior covariance matrix 
    """
    N, K = inputs.shape
    Phi = np.matrix(inputs)
    t = np.matrix(targets).reshape((N,1))
    m0 = np.matrix(m0).reshape((K,1))
    S0_inv = np.matrix(np.linalg.inv(S0))
    SN = np.linalg.inv(S0_inv + beta*Phi.transpose()*Phi)
    mN = SN*(S0_inv*m0 + beta*Phi.transpose()*t)
    return np.array(mN).flatten(), np.array(SN)

def predictive_distribution(inputs, beta, mN, SN):
    """
    Calculates the predictive distribution a linear model. This amounts to a
    mean and variance for each input point.

    parameters
    ----------
    inputs - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    beta - the known noise precision
    mN - posterior mean of the weights (vector) 1d-array (or array-like)
        of length K
    SN - the posterior covariance matrix for the weights 2d (K x K)-array 

    returns
    -------
    ys - a vector of mean predictions, one for each input datapoint
    sigma2Ns - a vector of variances, one for each input data-point 
    """
    N, K = inputs.shape
    Phi = np.matrix(inputs)
    mN = np.matrix(mN).reshape((K,1))
    SN = np.matrix(SN)
    ys = Phi*mN
    # create an array of the right size with the uniform term
    sigma2Ns = np.ones(N)/beta
    for n in range(N):
        # now calculate and add in the data dependent term
        # NOTE: I couldn't work out a neat way of doing this without a for-loop
        # NOTE: but if anyone can, then please share the answer.
        phi_n = Phi[n,:].transpose()
        sigma2Ns[n] += phi_n.transpose()*SN*phi_n
    return np.array(ys).flatten(), np.array(sigma2Ns)

def construct_polynomial_approx(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        expanded_xs = np.matrix(expand_to_monomials(xs, degree))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function

def construct_feature_mapping_approx(feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        designmtx = np.matrix(feature_mapping(xs))
        return linear_model_predict(designmtx, weights)
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function

def construct_knn_approx(inputs, targets, k, distance=None):
    """
    For 1 dimensional training data, it produces a function:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.
    """
    inputs = np.resize(inputs, (1,inputs.size))
    def prediction_function(inputs):
        inputs = inputs.reshape((inputs.size,1))
        distances = distance(inputs, inputs)
        predicts = np.empty(inputs.size)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function


    

