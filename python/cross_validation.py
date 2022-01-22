#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

from validation.cross_validator import CrossValidator

from models.logistic_classification import LogisticClassifier
from models.knn import KNNClassifier
from models.ann import ANNClassifier

from McNemars_test import get_multiple_mcn_p, get_multiple_prediction_matrix
from Confusion_matrix import get_confusion_matrices, plot_conf_matrices
from validation.validation_result import ValidationResult


#%%
#Get dataset

def get_dataset(file_name, spatial_dimensions = (True, True, True)):
    #Load dataset
    dataset = np.load(file_name)
    
    #Create spatial dimensions mask
    spat_mask = np.array(spatial_dimensions)

    #Reshapes to (1600, 300), 
    # where first dimension is one trajectory,
    # and each 10 trajectories the person is changed
    # and each 100 trajectories the experiment is changed.
    # Where each trajectory is listed as x0, y0, z0, ..., x299, y299, z299
    X = dataset[:, :, :, :, spat_mask].reshape((reduce(np.multiply, dataset.shape[0:3]), -1))
    #One hot encode person
    # X_p_idx = np.tile(np.arange(10).repeat(10), 16)
    # X_p_data = np.zeros((X.shape[0], dataset.shape[1]))
    # X_p_data[np.arange(X.shape[0]), X_p_idx] = 1
    #Prepend person feature onto X
    # X = np.hstack((X_p_data, X))


    #One-indexed experiment number labels
    y = (np.arange(16) + 1).repeat(100)
    
    
    return (X, y)


X, y = get_dataset("./data_numpy.npy", spatial_dimensions=(True, True, True))
# X, y = get_dataset("./data_numpy.npy", spatial_dimensions=(True, True, False))
# X, y = get_dataset("./data_numpy.npy", spatial_dimensions=(True, False, True))
# X, y = get_dataset("./data_numpy.npy", spatial_dimensions=(False, True, True))


#Define loss function
def loss_fn(pred, label):
    #Accuracy
    return np.sum(pred == label) / len(label)

#Create cross validator
cv = CrossValidator(X, y, loss_fn, 
                    stratified=True, groups=(np.arange(10)+1).repeat(160),
                    n_inner=10, n_outer=0, 
                    n_workers=4, randomize_seed=9999, 
                    verbose=True,)

#%%
#Define tester
def test(models, name):
    #Cross validate
    result = cv.cross_validate(models)
    
    #Save to file
    with open(f"./results/{name}.p", "wb") as file:
        pickle.dump(result, file)
    
    #Return
    return result


#Test models
result_name = "xyz_results"



results = test([LogisticClassifier(),
                ANNClassifier(lr=1e-5, n_epochs=10**5, hidden_units=40),
                KNNClassifier(40, 1)],
                result_name)


#%%
plot_title = "YZ"
result_name = f"{plot_title.lower()}_results"
model_names = ["LogReg", "FFNN", "KNN"]

with open(f"./results/{result_name}.p", "rb") as file:
    results: ValidationResult = pickle.load(file)


preds = results.test_preds_inner.T
labels = results.test_labels_inner

#Confusion matrices
conf_matrices = get_confusion_matrices(labels, *preds)
plot_conf_matrices(conf_matrices, model_names, title = f"{plot_title} confusion matrices")

#%%



#Significance levels
_,p_dict = get_multiple_mcn_p(labels, *preds)

mcnemar_table = "\\begin{table}[] \n\\centering \n\\begin{tabular}{l|lllll}  \n & BH-adjusted $p$-value & $n_{12}$ & $n_{21}$ & $n_{11}$ & $n_{22}$ \\\\ \\hline\n"

# print("Prediction matrix:")
for k,v in get_multiple_prediction_matrix(labels, *preds).items():
    mcnemar_table += f"{model_names[k[0]]} vs {model_names[k[1]]} & {p_dict[k][1]:.3} & "
    mcnemar_table += " & ".join(v.astype(str))
    mcnemar_table += "\\\\ \n"
    # print(k)
    # print(v)

mcnemar_table += "\end{tabular} \n\label{tab: mcnemar}\n\end{table}"

#Accuracies

accuracies = np.sum(preds == labels, axis = 1) / len(labels)
accuracy_table = "\\begin{table}[] \n\\centering \n\\begin{tabular}{l|lll} \n"

accuracy_table += "Model & " + " & ".join(model_names) + "\\\\ \\hline\n"
accuracies_round = [str(np.round(acc*100,1)) + "\\%" for acc in accuracies]
accuracy_table += "Accuracy & " + " & ".join(accuracies_round) + "\\\\\n"
accuracy_table += "\end{tabular} \n\label{tab: accuracy}\n\end{table}"


print(mcnemar_table)
print()
print(accuracy_table)