import pyperclip
import numpy as np
from McNemars_test import get_multiple_mcn_p, get_multiple_prediction_matrix, get_mcn_p, get_prediction_matrix
from Confusion_matrix import get_confusion_matrices, plot_conf_matrices
from validation.validation_result import ValidationResult



def get_latex_mcn_model_tables(plot_title, random_preds, model_names):
    result_name = f"{plot_title.lower()}_results"

    with open(f"./results/{result_name}.p", "rb") as file:
        results: ValidationResult = pickle.load(file)

    preds = results.test_preds_inner.T
    labels = results.test_labels_inner

    #Significance levels
    _,p_dict = get_multiple_mcn_p(labels, *preds, random_preds)

    mcnemar_table = f"\n% using coordinates {plot_title}\n\n"

    mcnemar_table += "\\begin{table}[H] \n\\centering \n\\begin{tabular}{l|lllll}  \n & BH-adjusted $p$-value & $n_{12}$ & $n_{21}$ & $n_{11}$ & $n_{22}$ \\\\ \\hline\n"

    # print("Prediction matrix:")
    for k,v in get_multiple_prediction_matrix(labels, *preds, random_preds).items():
        mcnemar_table += f"{model_names[k[0]]} vs {model_names[k[1]]} & {p_dict[k][1]:.3} & "
        mcnemar_table += " & ".join(v.astype(str))
        mcnemar_table += "\\\\ \n"
        # print(k)
        # print(v)

    mcnemar_table += "\end{tabular} \n"
    mcnemar_table += f"\label{{tab: mcnnemar {plot_title.lower()}}}\n"
    mcnemar_table += f"\caption{{McNemar test when using coordinates {plot_title.lower()}}}\n"
    mcnemar_table += "\end{table}"

    return mcnemar_table


def get_latex_mcn_spatial_tables(model_names):
    result_name = "xyz_results"
    plot_titles = ["xy","xz","yz"]

    n_comparisons = len(model_names)*len(plot_titles)

    p_values = np.zeros((n_comparisons, 8))

    with open(f"./results/{result_name}.p", "rb") as file:
        results_xyz: ValidationResult = pickle.load(file)
    
    preds_xyz = results_xyz.test_preds_inner.T
    labels_xyz = results_xyz.test_labels_inner

    i = 0
    for coordinate, plot_title in enumerate(plot_titles):
        result_name = f"{plot_title}_results"
        with open(f"./results/{result_name}.p", "rb") as file:
            results_ab: ValidationResult = pickle.load(file)

        preds_ab = results_ab.test_preds_inner.T
        labels_ab = results_ab.test_labels_inner

        if not np.all(labels_ab == labels_xyz): raise ValueError("Oh nooooo")

        for model, model_name in enumerate(model_names):
            n12,n21,n11,n22 = get_prediction_matrix(labels_xyz, preds_xyz[model], preds_ab[model])
            p_value = get_mcn_p(labels_xyz, preds_xyz[model], preds_ab[model])
            p_values[i] = [model, coordinate, p_value, 0, n12,n21,n11,n22]
            i += 1
    
    # BH correction
    p_values = p_values[np.argsort(p_values[:,2])]
    p_values[-1,3] = min(p_values[-1,2],1)

    for i in reversed(range(n_comparisons-1)):
        p_values[i,3] = min(p_values[i,2]*n_comparisons/(i+1), p_values[i+1,3])
    
    mask = np.argsort(p_values[:,0]*10 + p_values[:,1])
    p_values = p_values[mask]


    mcn_coordinate_table = ["\n% McNemar comparing different spatial coordinates\n\n"]
    mcn_coordinate_table.append("\\begin{table}[H] \n\\centering \n\\begin{tabular}{ll|lllll}\n")
    mcn_coordinate_table.append("Model & xyz vs. & $p$-value & $n_{12}$ & $n_{21}$ & $n_{11}$ & $n_{22}$ \\\\")

    prev_model = -1
    for [model,coordinate,_,BH_p_value, n12,n21,n11,n22] in p_values:
        model,coordinate, n12,n21,n11,n22 = tuple(map(int, [model,coordinate, n12,n21,n11,n22]))

        if prev_model != model:
            model_name = model_names[model]
            mcn_coordinate_table.append("\\hline \n")
            prev_model = model
        else:
            model_name = "  "
            mcn_coordinate_table.append("\n")

        mcn_coordinate_table.append(f"{model_name} & {plot_titles[coordinate]} & {BH_p_value:.3} & {n12} & {n21} & {n11} & {n22}\\\\ \n")
    
    mcn_coordinate_table.append("\end{tabular} \n")
    mcn_coordinate_table.append("\label{tab: mcnemar spatial}\n")
    mcn_coordinate_table.append("\caption{McNemar test comparing same model using different spatial coordinates}\n")
    mcn_coordinate_table.append("\end{table}")

    return "".join(mcn_coordinate_table)



def get_latex_acc_table(plot_titles, random_preds, model_names):

    ls = "l"*len(model_names)

    accuracy_table = "\n% Accuracy table\n"

    accuracy_table += f"\\begin{{table}}[H] \n\\centering \n\\begin{{tabular}}{{l|{ls}}} \n"
    accuracy_table += "Axis & " + " & ".join(model_names) + "\\\\ \\hline\n"
    
    for plot_title in plot_titles:
        
        result_name = f"{plot_title.lower()}_results"

        with open(f"./results/{result_name}.p", "rb") as file:
            results: ValidationResult = pickle.load(file)

        preds = results.test_preds_inner.T
        labels = results.test_labels_inner

        #Accuracies
        accuracies = np.sum(preds == labels, axis = 1) / len(labels)
        accuracies = np.append(accuracies, np.sum(labels==random_preds)/len(labels))
        accuracies_round = [str(np.round(acc*100,1)) + "\\%" for acc in accuracies]
        accuracy_table += f"{plot_title.lower()} & " + " & ".join(accuracies_round) + "\\\\\n"
    
    accuracy_table += "\end{tabular} \n"
    accuracy_table += "\label{tab: accuracy all}\n"
    accuracy_table += "\caption{Accuracies of different models using different spatial coordinates}\n"
    accuracy_table += "\end{table}"

    return accuracy_table



model_names = ["Multi. Reg", "FFNN", "KNN", "Baseline"]
plot_titles = ["XYZ","XY","XZ","XY"]

output = []

output.append(get_latex_mcn_spatial_tables(model_names[:-1]))

output.append(get_latex_acc_table(plot_titles, random_preds, model_names))

for plot_title in plot_titles:
    output.append(get_latex_mcn_model_tables(plot_title, random_preds, model_names))


pyperclip.copy("\n".join(output))


