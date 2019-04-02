import numpy as np
import pandas as pd

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from fomlads.evaluate.partition import train_and_test_filter
from fomlads.evaluate.partition import train_and_test_partition
from fomlads.evaluate.partition import create_cv_folds
from fomlads.model.basis_functions import quadratic_feature_mapping

# machine learning
from sklearn.neighbors import KNeighborsClassifier



def main(dataset):

    inputs,targets,label = process_data(dataset)
    N = inputs.shape[0]  # Total number of datasets
    num_knn = 30         # number of nearest neighbours + 1
    num_folds = 5        # number of folds

    #Partitioning train and test data

    train_filter, test_filter = train_and_test_filter(N, test_fraction=0.20)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter,
                                                                                      test_filter)
    cv_folds = create_cv_folds(N, num_folds)


    #Cross validations, obtaining best parameter for KNN

    optimumKNN,neighbors,mean_validation_accuracy,train_accuracy = Finding_most_optimum_k(inputs,targets,num_knn,cv_folds,num_folds)
    print("The best mean validation score = {} with number of nearest neighbours = {}".format(mean_validation_accuracy[optimumKNN],
                                                                                              optimumKNN))

    #Model fitting with the optimum number of neighbors

    knn,fitting_accuracy,prediction_accuracy = fitting_best_k(optimumKNN, train_inputs, train_targets, test_inputs, test_targets)


    #Preparation for ROC curve

    fpr, tpr, AUC = ROC_values_and_AUC(test_inputs, test_targets, 100,knn)
    
    #QUADRATIC EXPANSION
    quadratic_inputs = quadratic_feature_mapping(inputs)
    quadratic_train_inputs = quadratic_feature_mapping(train_inputs)
    quadratic_test_inputs = quadratic_feature_mapping(test_inputs)
    quadratic_optimumKNN,neighbors,quadratic_mean_validation_accuracy,quadratic_train_accuracy = Finding_most_optimum_k(quadratic_inputs,targets,num_knn,cv_folds,num_folds)
    print("The best mean validation score = {} with number of nearest neighbours = {}".format(quadratic_mean_validation_accuracy[quadratic_optimumKNN],
                                                                                              quadratic_optimumKNN))

    #Model fitting with the optimum number of neighbors

    quadratic_knn,quadratic_fitting_accuracy,quadratic_prediction_accuracy = fitting_best_k(quadratic_optimumKNN, quadratic_train_inputs, train_targets, quadratic_test_inputs, test_targets)


    #Preparation for ROC curve

    quad_fpr, quad_tpr, quad_AUC = ROC_values_and_AUC(quadratic_test_inputs, test_targets, 100,quadratic_knn)



    #Plotting testing accuracy and training accuracy against no. of neighbors to see if the model overfit

    fig = plt.figure()
    fig.suptitle("Accuracy for different K")
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(neighbors, mean_validation_accuracy, label='Testing Accuracy')
    ax1.plot(neighbors, train_accuracy, label='Training Accuracy')
    ax1.set_xlabel('Number of Neighbors')
    ax1.set_ylabel('Accuracy')
    ax1.plot(neighbors, quadratic_mean_validation_accuracy, label='Quadratic Testing Accuracy')
    ax1.plot(neighbors, quadratic_train_accuracy, label='Quadratic Training Accuracy')
    ax1.legend()


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



    #Constructing confusion matrix
    y_pred = knn.predict(test_inputs)
    y_actual = test_targets
    confusion_matrix = pd.crosstab(y_pred, y_actual).T.as_matrix()
    confusion_matrix = pd.DataFrame(confusion_matrix, index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    fig3.suptitle("Normal Confusion Matrix")
    sns.heatmap(confusion_matrix, annot=True, linewidths=0.3, linecolor="White", cbar=False, fmt=".0f", ax=ax3,
                cmap="Blues")
    ax3.set_xlabel("Predicted class")
    ax3.set_ylabel("Actual class")


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


    y_pred_quadratic = quadratic_knn.predict(quadratic_test_inputs)
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
    
    # Creating a figure with all the numbers
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(2, 2, 1)
    ax4.text(0, 1.0, 'Results',fontsize = 12,fontweight='bold')
    ax4.text(0, 0.7, 'Accuracy with no basis functions = {} with best k = {}'.format(round(mean_validation_accuracy[optimumKNN], 4), optimumKNN))
    ax4.text(0, 0.6, 'Accuracy with with quadratic basis functions = {} with best k = {}'.format(round(quadratic_mean_validation_accuracy[quadratic_optimumKNN], 4), quadratic_optimumKNN))
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




def Finding_most_optimum_k(inputs,targets,num_knn,cv_folds,num_folds):

    # create array of zeros to store the mean validation accuracy for k at location k-1
    mean_validation_accuracy = np.zeros(num_knn - 1)
    mean_train_accuracy = np.zeros(num_knn - 1)
    for f in range(num_folds):
        fold_filters = cv_folds[f]
        training_filter = fold_filters[0]
        validation_filter = fold_filters[1]

        training_data, training_targets, validation_data, validation_targets = train_and_test_partition(inputs, targets,
                                                                                                        training_filter,
                                                                                                        validation_filter)
        neighbors = np.arange(1, num_knn)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))

        # Loop over different values of k
        for i, k in enumerate(neighbors):
            # Setup a k-NN Classifier with k neighbors: knn
            knn = KNeighborsClassifier(n_neighbors=k)
            # Fit the classifier to the training data
            knn.fit(training_data, training_targets)

            # Compute accuracy on the training set
            train_accuracy[i] = knn.score(training_data, training_targets)
            # Compute accuracy on the testing set
            test_accuracy[i] = knn.score(validation_data, validation_targets)
            mean_validation_accuracy[i] += test_accuracy[i] / num_folds
            mean_train_accuracy[i] += train_accuracy[i] / num_folds

    optimumKNN = test_accuracy.tolist().index(np.max(test_accuracy)) + 1
    return optimumKNN,neighbors,mean_validation_accuracy,mean_train_accuracy


def fitting_best_k(optimumKNN,train_inputs,train_targets,test_inputs,test_targets):

    # Create a k-NN classifier with optimum number of neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=optimumKNN)
    # Fit the classifier to the training data and predict on test data
    knn.fit(train_inputs, train_targets)
    knn.predict(test_inputs)
    # Print the accuracy
    print('KNN Fitting accuracy' + "\n" + '**************************')
    fitting_accuracy = knn.score(train_inputs, train_targets)
    print(knn.score(train_inputs, train_targets))
    print('KNN Prediction accuracy' + "\n" + '**************************')
    prediction_accuracy = knn.score(test_inputs, test_targets)
    print(knn.score(test_inputs, test_targets))
    return knn,fitting_accuracy, prediction_accuracy

def process_data(dataset):

#    The code below that was commented was used when
#    we were making the size of abalone's dataset = the size of Titanic's dataset
#    data = data.sample (n=891, frac=None, replace=False, weights=None, random_state=None, axis=None)

    data = pd.read_csv("{}".format(dataset))
    print('This dataset has {} observations with {} features.'.format(data.shape[0], data.shape[1]))
    inputs_header = data.columns[:-1]
    inputs = data[inputs_header].values
    label = "{}".format(data.columns[-1])
    targets = data[label].values
    return inputs, targets,label


def ROC_values_and_AUC(test_inputs, test_targets, num_points,knn):
    AUC = 0.0
    false_positive_rates = np.empty(num_points)
    true_positive_rates = np.empty(num_points)
    num_neg = np.sum(1 - test_targets)
    num_pos = np.sum(test_targets)
    prediction_probs = knn.predict_proba(test_inputs)
    thresholds = np.linspace(-0.001, 1.001, num_points)
    for i in range(num_points):
        predicts = (prediction_probs[:, 1] > thresholds[i]).astype(int)
        num_false_positives = np.sum((predicts == 1) & (test_targets == 0))
        num_true_positives = np.sum((predicts == 1) & (test_targets == 1))
        false_positive_rates[i] = (num_false_positives) / num_neg
        true_positive_rates[i] = (num_true_positives) / num_pos

    AUC = -np.trapz(true_positive_rates, false_positive_rates)
    print("AUC = ", AUC)
    return false_positive_rates, true_positive_rates, AUC

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        raise IOError("Please run the file with the dataset as the parameter.")
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(sys.argv[1])
    else:
        raise IOError("Too many arguments, please include only the dataset.")