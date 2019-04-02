import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from fomlads.plot.evaluations import plot_roc
from fomlads.evaluate.eval_classification import cross_entropy_error
from fomlads.model.classification import shared_covariance_model_fit, shared_covariance_model_predict,logistic_regression_fit,logistic_regression_predict, logistic_regression_prediction_probs
from fomlads.model.basis_functions import quadratic_feature_mapping
from fomlads.evaluate.partition import train_and_test_filter, train_and_test_partition, create_cv_folds


def main(dataset):
    inputs,targets,label = process_data(dataset)
    quadratic_inputs = quadratic_feature_mapping(inputs)
    N = inputs.shape[0]  # Total number of datasets
    num_folds = 5        # number of folds
    #Partitioning train and test data


    cv_folds = create_cv_folds(N, num_folds)

    #Cross validations, obtaining best parameter for KNN
    if ("titanic" in dataset.lower()):
        print("No cross validation is done on the titanic data because its run time exceeds 45 minutes")
        print("Please wait...")
        normal_test_accuracy, normal_entropy = logistic_fit_and_analysis_no_folds(inputs,targets, 1e-5, 1e-3)
        quadratic_test_accuracy, quadratic_entropy = logistic_fit_and_analysis_no_folds(quadratic_inputs,targets, 1e-5, 1e-3)
    else:
        print("Cross validation is running")
        print("Please wait...")
        normal_test_accuracy, normal_entropy = logistic_fit_and_analysis(inputs,targets,cv_folds,num_folds, 1e-6, 1e-6)
        quadratic_test_accuracy, quadratic_entropy = logistic_fit_and_analysis(quadratic_inputs,targets,cv_folds,num_folds, 1e-6, 1e-6)
    print("The accuracy = {} and cross entropy error = {} for no basis functions".format(normal_test_accuracy, normal_entropy))
    print("The accuracy = {} and cross entropy error = {} for a quadratic basis functions".format(quadratic_test_accuracy, quadratic_entropy))

    train_filter, test_filter = train_and_test_filter(N, test_fraction=0.20)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter,
                                                                                    test_filter)
    
    quadratic_train_inputs = quadratic_feature_mapping(train_inputs)
    quadratic_test_inputs = quadratic_feature_mapping(test_inputs)
    
    weights = robust_logistic_regression_fit(train_inputs, train_targets, 1e-5, 1e-3)
    quadratic_weights = robust_logistic_regression_fit(quadratic_train_inputs, train_targets, 1e-5, 1e-3)
    num_points = 500
    print("No Basis Function ROC")
    fpr, tpr, AUC = ROC_values_and_AUC(test_inputs, test_targets, weights, num_points)
    print("Quadratic Expansion Basis Function ROC")
    quad_fpr, quad_tpr, quad_AUC = ROC_values_and_AUC(quadratic_test_inputs, test_targets, quadratic_weights, num_points)    


    #Plotting ROC curve

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(fpr, tpr, '-', color = "b", label='Normal AUC = %0.4f' % AUC)
    ax2.plot(quad_fpr, quad_tpr, '-', color = "r", label='Quadratic AUC = %0.4f' % quad_AUC)
    ax2.legend(loc='lower right')
    ax2.plot([0, 1], [0, 1], linestyle='--')    
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim([-0.01,1.01])
    ax2.set_ylim([-0.01,1.01])
    ax2.set_xticks([0,0.5,1])
    ax2.set_yticks([0,0.5,1])
    plt.tight_layout()

    y_pred = logistic_regression_predict(test_inputs, weights)
    y_actual = test_targets
    confusion_matrix = pd.crosstab(y_pred,y_actual).T.as_matrix()

    confusion_matrix = pd.DataFrame(confusion_matrix, index = ['Negative', 'Positive'], columns = ['Negative', 'Positive'])

    fig3, ax3= plt.subplots(figsize=(5,5))
    fig3.suptitle("Normal Confusion Matrix")
    sns.heatmap(confusion_matrix,annot=True,linewidths=0.3,linecolor="White",cbar=False,fmt=".0f",ax=ax3,cmap="Blues")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")


    #Calculating performance of the model
    confusion_matrix = pd.crosstab(y_pred, y_actual).T.as_matrix()
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]

    Precision = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    F1Score = 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
    print('Precision   = ', Precision)
    print('Sensitivity = ', Sensitivity)
    print('Specificity = ', Specificity)
    print('F1 Score    = ', F1Score)
    
    
    y_pred_quadratic = logistic_regression_predict(quadratic_test_inputs, quadratic_weights)
    quadratic_confusion_matrix = pd.crosstab(y_pred_quadratic,y_actual).T.as_matrix()

    quadratic_confusion_matrix = pd.DataFrame(quadratic_confusion_matrix, index = ['Negative', 'Positive'], columns = ['Negative', 'Positive'])

    fig5, ax5= plt.subplots(figsize=(5,5))
    fig5.suptitle("Quadratic Confusion Matrix")
    sns.heatmap(quadratic_confusion_matrix,annot=True,linewidths=0.3,linecolor="White",cbar=False,fmt=".0f",ax=ax5,cmap="Blues")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    
    print("made it this far")

    #Calculating performance of the model
    quadratic_confusion_matrix = pd.crosstab(y_pred_quadratic, y_actual).T.as_matrix()
    print("made the confusion matrix")
    TN_quadratic = quadratic_confusion_matrix[0, 0]
    FN_quadratic = quadratic_confusion_matrix[1, 0]
    TP_quadratic = quadratic_confusion_matrix[1, 1]
    FP_quadratic = quadratic_confusion_matrix[0, 1]

    Precision_quadratic = TP_quadratic / (TP_quadratic + FP_quadratic)
    Sensitivity_quadratic = TP_quadratic / (TP_quadratic + FN_quadratic)
    Specificity_quadratic = TN_quadratic / (TN_quadratic + FP_quadratic)
    F1Score_quadratic = 2 * ((Precision_quadratic * Sensitivity_quadratic) / (Precision_quadratic + Sensitivity_quadratic))
    print('Precision   = ', Precision_quadratic)
    print('Sensitivity = ', Sensitivity_quadratic)
    print('Specificity = ', Specificity_quadratic)
    print('F1 Score    = ', F1Score_quadratic)

    #Creating a figure with the results
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(2, 2, 1)
    ax4.text(0, 1.2, 'Results',fontsize = 12,fontweight='bold')
    if ("titanic" in dataset.lower()):
        ax4.text(0, 1.0, 'Accuracy with no basis functions = {}'.format(round(normal_test_accuracy, 4)))
        ax4.text(0, 0.9, 'Accuracy with with quadratic basis functions = {}'.format(round(quadratic_test_accuracy, 4)))
    else:
        ax4.text(0, 1.0, 'Mean accuracy with no basis functions = {}'.format(round(normal_test_accuracy, 4)))
        ax4.text(0, 0.9, 'Mean accuracy with with quadratic basis functions = {}'.format(round(quadratic_test_accuracy, 4)))
    ax4.text(0, 0.7, 'Cross Entropy Error with no basis functions = {}'.format(round(normal_entropy,4)))
    ax4.text(0, 0.6, 'Cross Entropy Error with quadratic basis functions = {}'.format(round(quadratic_entropy,4)))
    ax4.text(0, 0.4, 'Area Under Curves with no basis functions = {}'.format(round(AUC, 4)))
    ax4.text(0, 0.3, 'Area Under Curves with quadratic basis functions = {}'.format(round(quad_AUC, 4)))
    ax4.text(0, 0.1, 'Precision   = {}'.format(round(Precision, 4)))
    ax4.text(0, 0, 'Sensitivity   = {}'.format(round(Sensitivity, 4)))
    ax4.text(0, -0.1, 'Specificity   = {}'.format(round(Specificity, 4)))
    ax4.text(0, -0.2, 'F1 Score   = {}'.format(round(F1Score, 4)))
    ax4.text(0, -0.3, 'Quadratic Precision   = {}'.format(round(Precision_quadratic, 4)))
    ax4.text(0, -0.4, 'Quadratic Sensitivity   = {}'.format(round(Sensitivity_quadratic, 4)))
    ax4.text(0, -0.5, 'Quadratic Specificity   = {}'.format(round(Specificity_quadratic, 4)))
    ax4.text(0, -0.6, 'Quadratic F1 Score   = {}'.format(round(F1Score_quadratic, 4)))
    ax4.axis('off')
    
    #Showing all plots
    plt.show()

def logistic_fit_and_analysis(inputs,targets,cv_folds,num_folds, lambda_val, threshold):
    accuracy_array = np.zeros(num_folds)
    entropy_array = np.zeros(num_folds)
    for f in range(num_folds):
        fold_filters = cv_folds[f]
        training_filter = fold_filters[0]
        validation_filter = fold_filters[1]
        training_data, training_targets, validation_data, validation_targets = train_and_test_partition(inputs, targets, training_filter, validation_filter)
        log_reg_weights = robust_logistic_regression_fit(training_data, training_targets, lambda_val, threshold=threshold)
        predicted = logistic_regression_predict(validation_data, log_reg_weights)
        #ACCURACY
        predict_probs = logistic_regression_prediction_probs(validation_data, log_reg_weights)
        #print(validation_data.shape, validation_targets.shape, predict_probs.shape)
        for i in range(len(validation_targets)):
            accuracy_array[f] += (predicted[i]==validation_targets[i]).astype(int)/len(validation_targets) 
            if (predict_probs[i] == 1):
                predict_probs[i] -= 0.0001 #THIS IS SO THAT IN CROSS_ENTROPY_ERROR THERE IS NEVER A LOG OF 0 
        entropy_array[f] = cross_entropy_error(validation_targets, predict_probs)     
    accuracy = accuracy_array.mean()
    entropy = entropy_array.mean()
    return accuracy, entropy

def logistic_fit_and_analysis_no_folds(inputs,targets,lambda_val, threshold):
    N = inputs.shape[0]
    training_filter, validation_filter = train_and_test_filter(N, test_fraction=0.20)
    training_data, training_targets, validation_data, validation_targets = train_and_test_partition(inputs, targets, training_filter, validation_filter)
    log_reg_weights = robust_logistic_regression_fit(training_data, training_targets, lambda_val, threshold=threshold)
    predicted = logistic_regression_predict(validation_data, log_reg_weights)
    #ACCURACY
    predict_probs = logistic_regression_prediction_probs(validation_data, log_reg_weights)
    #print(validation_data.shape, validation_targets.shape, predict_probs.shape)
    
    accuracy = 0.0
    for i in range(len(validation_targets)):
        accuracy += (predicted[i]==validation_targets[i]).astype(int)/len(validation_targets)
        if (predict_probs[i] == 1):
            predict_probs[i] -= 0.0001 #THIS IS SO THAT IN CROSS_ENTROPY_ERROR THERE IS NEVER A LOG OF 0
    entropy = cross_entropy_error(validation_targets, predict_probs)       
    return accuracy, entropy 

def process_data(dataset):
    data = pd.read_csv("{}".format(dataset))
    print('This dataset has {} observations with {} features.'.format(data.shape[0], data.shape[1]))
    inputs_header = data.columns[:-1]
    inputs = data[inputs_header].values
    label = "{}".format(data.columns[-1])
    targets = data[label].values
    return inputs, targets,label

def ROC_values_and_AUC(inputs, targets, weights, num_points):
    false_positive_rates = np.empty(num_points)
    true_positive_rates = np.empty(num_points)
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)    
    thresholds = np.linspace(-0.001,1.001,num_points)
    for i in range(num_points):
        prediction_probs = logistic_regression_prediction_probs(inputs, weights)
        predicts = (prediction_probs > thresholds[i]).astype(int)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        false_positive_rates[i] = (num_false_positives)/num_neg
        true_positive_rates[i] = (num_true_positives)/num_pos
    AUC = -np.trapz(true_positive_rates, false_positive_rates)
    print("AUC = ", AUC)
    return false_positive_rates, true_positive_rates, AUC


def robust_logistic_regression_fit(
        inputs, targets, lambda_val, threshold=1e-6, weights0=None):
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
        H = inputs.T*R*inputs 
        H += lambda_val*np.identity(D)
        H_inv = np.linalg.inv(H)
        # update the weights
        new_weights = weights - H_inv*inputs.T*np.matrix(predicts-targets)
        # calculate the update_magnitude
        update_magnitude = np.sqrt(np.sum(np.array(new_weights-weights)**2))
        #print(update_magnitude)
        # update the weights
        weights = new_weights
    return weights

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        raise IOError("Please run the file with the dataset as the parameter.")
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(sys.argv[1])
    else:
        raise IOError("Too many arguments, please include only the dataset.")