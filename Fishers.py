import numpy as np
import pandas as pd

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

from fomlads.model.classification import project_data,fisher_linear_discriminant_projection, robust_fisher_linear_discriminant_projection
from fomlads.model.basis_functions import quadratic_feature_mapping
from fomlads.evaluate.partition import train_and_test_filter, train_and_test_partition, create_cv_folds
import time

def main(dataset):
    num_folds = 5   # number of folds
    inputs,targets,label = process_data(dataset)
    N = inputs.shape[0]  # total number of datasets
    if ('titanic' in dataset.lower()):
        name = 'Titanic'
        num_decision_boundaries_normal = 50
        decision_boundaries_normal = np.linspace(-1, 1, num_decision_boundaries_normal)
    else:
        name = 'Abalone'
        num_decision_boundaries_normal = 30
        decision_boundaries_normal = np.linspace(0.1,0.3, num_decision_boundaries_normal)


    #Partitioning train and test data

    train_filter, test_filter = train_and_test_filter(N, test_fraction=0.3)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter,
                                                                                      test_filter)
    cv_folds = create_cv_folds(N, num_folds)

    #Cross validation
    train_accuracy_array, test_accuracy_array, decision_boundary = cross_validation_decision_boundary_fishers(inputs, targets, cv_folds, num_folds, decision_boundaries_normal, num_decision_boundaries_normal,0)
    weights = fisher_linear_discriminant_projection(train_inputs, train_targets)
    predicted = predict(test_inputs, weights, decision_boundary)
    
    #Cross Validation for quadratic
    pol2_inputs = quadratic_feature_mapping(inputs)
    pol2_train_inputs = quadratic_feature_mapping(train_inputs)
    pol2_test_inputs = quadratic_feature_mapping(test_inputs)
    num_decision_boundaries_robust = 30
    decision_boundaries_robust = np.linspace(-0.1, 0.1, num_decision_boundaries_robust)
    
    quadratic_train_accuracy_array, quadratic_test_accuracy_array, quadratic_decision_boundary =cross_validation_decision_boundary_fishers(pol2_inputs, targets, cv_folds, num_folds, decision_boundaries_robust,num_decision_boundaries_robust, 1)
    quadratic_weights = robust_fisher_linear_discriminant_projection(pol2_train_inputs, train_targets,1e-6)
    quadratic_predicted = predict(pol2_test_inputs, quadratic_weights, quadratic_decision_boundary)

    #Preparation for AUC plot
    false_positive_rates, true_positive_rates, AUC = ROC_values_and_AUC(train_inputs, train_targets, test_inputs,test_targets, 0)

    pol2_false_positive_rates, pol2_true_positive_rates, pol2_AUC = ROC_values_and_AUC(pol2_train_inputs, train_targets, pol2_test_inputs,test_targets, 1)

    #Plotting testing accuracy and training accuracy on changing decision boundaries
    #Normal data
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_title('Fisher, changing decision boundaries, {}'.format(name))
    ax1.plot(-decision_boundaries_normal, test_accuracy_array, label='Testing Accuracy')
    ax1.plot(-decision_boundaries_normal, train_accuracy_array, label='Training Accuracy')
    ax1.legend()
    ax1.set_xlabel('Decision Boundary')
    ax1.set_ylabel('Accuracy')

    #Transformed data (Quadratic)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_title('Fisher, changing decision boundaries, Quadratic, {}'.format(name))
    ax2.plot(decision_boundaries_robust, quadratic_test_accuracy_array, label='Testing Accuracy')
    ax2.plot(decision_boundaries_robust, quadratic_train_accuracy_array, label='Training Accuracy')
    ax2.legend()
    ax2.set_xlabel('Decision Boundary')
    ax2.set_ylabel('Accuracy')


    #Plotting ROC curve
    fig3 = plt.figure(figsize=(6, 6))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(false_positive_rates, true_positive_rates,'-' ,color="b",label='AUC normal = %0.2f' % AUC)
    ax3.plot(pol2_false_positive_rates, pol2_true_positive_rates,'-', color="r",label='AUC quadratic = %0.2f' % pol2_AUC)
    ax3.legend(loc='lower right')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_aspect('equal', 'box')
    ax3.plot([0, 1], [0, 1], linestyle='--')
    ax3.set_xlim([-0.01, 1.01])
    ax3.set_ylim([-0.01, 1.01])
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_yticks([0, 0.5, 1])
    plt.tight_layout()
    print("The AUC with no basis function = ", AUC)
    print("The AUC with quadratic expansion = ", pol2_AUC)




    weights = fisher_linear_discriminant_projection(train_inputs, train_targets)
    predicted = predict(test_inputs, weights, decision_boundary)
    y_pred = predicted
    y_actual = test_targets
    try:
        confusion_matrix = pd.crosstab(y_pred, y_actual).T.as_matrix()
        confusion_matrix = pd.DataFrame(confusion_matrix, index = ['Negative', 'Positive'],columns = ['Negative', 'Positive'] )
    except:
        print('Sorry, Pandas is acting weird (trust me), please run the program again.')
        exit(0)



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
    
    quadratic_weights = robust_fisher_linear_discriminant_projection(pol2_train_inputs, train_targets,1e-6)
    quadratic_predicted = predict(pol2_test_inputs, quadratic_weights, quadratic_decision_boundary)
    y_pred_quadratic = quadratic_predicted
    quadratic_confusion_matrix = pd.crosstab(y_pred_quadratic,y_actual).T.as_matrix()

    quadratic_confusion_matrix = pd.DataFrame(quadratic_confusion_matrix, index = ['Negative', 'Positive'], columns = ['Negative', 'Positive'])

    fig5, ax5= plt.subplots(figsize=(5,5))
    fig5.suptitle("Quadratic Confusion Matrix")
    sns.heatmap(quadratic_confusion_matrix,annot=True,linewidths=0.3,linecolor="White",cbar=False,fmt=".0f",ax=ax5,cmap="Blues")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    
    print("made it this far")

    #Calculating performance of thee model
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
    ax4.text(0, 0.7, 'Results',fontsize = 12,fontweight='bold')
    ax4.text(0, 0.4, 'Area Under Curves with no basis functions = {}'.format(round(AUC, 4)))
    ax4.text(0, 0.3, 'Area Under Curves with quadratic basis functions = {}'.format(round(pol2_AUC, 4)))
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


def cross_validation_decision_boundary_fishers(inputs,targets,cv_folds,num_folds,decision_boundaries,num_decision_boundaries,robust=0):

    train_accuracy_array = np.zeros(num_decision_boundaries)
    test_accuracy_array = np.zeros(num_decision_boundaries)

    for f in range(num_folds):
        fold_filters = cv_folds[f]
        training_filter = fold_filters[0]
        validation_filter = fold_filters[1]
        training_data, training_targets, validation_data, validation_targets = train_and_test_partition(inputs, targets,
                                                                                                        training_filter,
                                                                                                        validation_filter)
        if robust == 0:
            fisher_weights = fisher_linear_discriminant_projection(training_data, training_targets)
        elif robust == 1:
            fisher_weights = robust_fisher_linear_discriminant_projection(training_data, training_targets, 1e-6)

        projected_inputs_train = project_data(training_data, fisher_weights)
        projected_inputs_test = project_data(validation_data, fisher_weights)
        new_ordering_train = np.argsort(projected_inputs_train)
        new_ordering_test = np.argsort(projected_inputs_test)
        projected_inputs_train = projected_inputs_train[new_ordering_train]
        projected_inputs_test = projected_inputs_test[new_ordering_test]

        training_targets = np.copy(training_targets[new_ordering_train])
        validation_targets = np.copy(validation_targets[new_ordering_test])
        predicted_train = np.empty(len(projected_inputs_train))
        predicted_test = np.empty(len(projected_inputs_test))


        for j in range(len(decision_boundaries)):
            train_accuracy_temp = 0.0
            test_accuracy_temp = 0.0
            for i in range(len(training_targets)):
                predicted_train[i] = (projected_inputs_train[i] > decision_boundaries[j]).astype(int)
                train_accuracy_temp += (predicted_train[i] == training_targets[i]).astype(int) / len(training_targets)

            for t in range(len(validation_targets)):
                predicted_test[t] = (projected_inputs_test[t] > decision_boundaries[j]).astype(int)
                test_accuracy_temp += (predicted_test[t] == validation_targets[t]).astype(int) / len(validation_targets)
            test_accuracy_array[j] += test_accuracy_temp / num_folds
            train_accuracy_array[j] += train_accuracy_temp / num_folds



    return train_accuracy_array,test_accuracy_array, decision_boundaries[test_accuracy_array.tolist().index(np.max(test_accuracy_array))]

def process_data(dataset):
    data = pd.read_csv("{}".format(dataset))
    print('This dataset has {} observations with {} features.'.format(data.shape[0], data.shape[1]))
    inputs_header = data.columns[:-1]
    inputs = data[inputs_header].values
    label = "{}".format(data.columns[-1])
    targets = data[label].values
    return inputs, targets,label

def predict(inputs, weights, decision_boundary):
    projected_inputs = project_data(inputs, weights)
    predicted = np.array(projected_inputs)
    for i, val in enumerate(projected_inputs):
        predicted[i]=(val>decision_boundary).astype(int)
    return predicted

def ROC_values_and_AUC(train_inputs,train_targets,test_inputs,test_targets, robust):
    if robust == 0:
        weights = fisher_linear_discriminant_projection(train_inputs, train_targets)
    elif robust == 1:
        weights = robust_fisher_linear_discriminant_projection(train_inputs, train_targets,1e-6)

    projected_inputs = project_data(test_inputs, weights)
    new_ordering = np.argsort(projected_inputs)
    projected_inputs = projected_inputs[new_ordering]
    plot_targets = np.copy(test_targets[new_ordering])
    N = test_targets.size
    num_neg = np.sum(1 - test_targets)
    num_pos = np.sum(test_targets)
    false_positive_rates = np.empty(N)
    true_positive_rates = np.empty(N)
    for i, w0 in enumerate(projected_inputs):
        false_positive_rates[i] = np.sum(1 - plot_targets[i:]) / num_neg
        true_positive_rates[i] = np.sum(plot_targets[i:]) / num_pos

    AUC = -np.trapz(true_positive_rates, false_positive_rates)

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