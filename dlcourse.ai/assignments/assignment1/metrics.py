import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(prediction)):
        if ground_truth[i] == prediction[i] == True:
            TP = TP + 1
        
    for i in range(len(prediction)):
        if ground_truth[i] == False and prediction[i] == True:
            FP = FP + 1

    for i in range(len(prediction)):
        if ground_truth[i] == True and prediction[i] == False :
            FN = FN + 1

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    f1 = 2*((precision*recall)/(precision+recall))

    num_samples = len(prediction)
    accuracy = np.sum(prediction == ground_truth) / num_samples   
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = 0
    num_samples = len(prediction)
    accuracy = np.sum(prediction == ground_truth) / num_samples    
    return accuracy
