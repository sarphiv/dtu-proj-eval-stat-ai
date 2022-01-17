
import numpy as np
from scipy.stats import binom


def get_prediction_matrix(true_labels, pred1, pred2):
    n11 = np.sum((true_labels == pred1) & (true_labels == pred2))
    n12 = np.sum((true_labels == pred1) & (true_labels != pred2))
    n21 = np.sum((true_labels != pred1) & (true_labels == pred2))
    n22 = np.sum((true_labels != pred1) & (true_labels != pred2))

    return np.array([n12, n21, n11, n22])


def get_multiple_prediction_matrix(true, *preds):
    results = {}
    
    for i in range(len(preds)-1):
        for j in range(i + 1, len(preds)):
            results[(i, j)] = get_prediction_matrix(true, preds[i], preds[j])
    
    return results


def get_mcn_p(true_labels, pred1, pred2):
    '''
    one mc-nemar test.
    computes p-value for one comparison
    inputs should be 1 dimensional equal sized np.arrays'''
    
    n12 = np.sum((true_labels == pred1) & (true_labels != pred2))
    n21 = np.sum((true_labels != pred1) & (true_labels == pred2))

    p_value = 2*binom.cdf(min(n12,n21), n21+n12, 0.5)

    return p_value


def get_multiple_mcn_p(true, *preds):
    '''
    multiple mc-nemar tests
    
    gets true and prediction labels from multiple models,
    and computes the pairwize p-values and BH corrected p-values for each possible comparison

    inputs:
    input1: true labels
    input2...k+1: predictions of model 1..k 
    all should be 1-dim equal sized np.arrays
    
    outputs:
    output1: np.array shape (n_comparisons, 4) - each line is [model_i, model_j, p-value, BH-p-value]
    output2: dictionary: keys are tuples:  (model_i, model_j)
                         items are the corresponding (p-value, BH-p-value)
    
    '''

    n_models = len(preds)
    n_comparisons = int(n_models*(n_models-1)/2)

    #compute p-values
    p_values = np.zeros((n_comparisons,4))

    i = 0
    for model1 in range(n_models-1):
        for model2 in range(model1+1,n_models):
            p_values[i,0:3] = [model1, model2, get_mcn_p(true, preds[model1], preds[model2])]
            i += 1
    
    # BH correction
    p_values = p_values[np.argsort(p_values[:,2])]
    p_values[-1,3] = min(p_values[-1,2],1)

    for i in reversed(range(n_comparisons-1)):
        p_values[i,3] = min(p_values[i,2]*n_comparisons/(i+1), p_values[i+1,3])
    
    #put into dict
    p_values_dict = {}
    for [m1,m2,p,pBH] in p_values:
        p_values_dict[int(m1),int(m2)] = p, pBH
        p_values_dict[int(m2),int(m1)] = p, pBH

    return p_values, p_values_dict 
