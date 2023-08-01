from sklearn.metrics import roc_auc_score


def macro_averaged_auc_roc_eval(model, dataset):
    """
    Let's assume you have three labels (A, B, C) and the true labels and predicted probabilities for five instances are as follows:

    Instance 1: True labels = [1, 0, 1], Predicted probabilities = [0.9, 0.3, 0.8]
    Instance 2: True labels = [0, 1, 1], Predicted probabilities = [0.2, 0.7, 0.6]
    Instance 3: True labels = [1, 1, 0], Predicted probabilities = [0.6, 0.4, 0.1]
    Instance 4: True labels = [1, 0, 0], Predicted probabilities = [0.7, 0.2, 0.3]
    Instance 5: True labels = [0, 1, 1], Predicted probabilities = [0.1, 0.6, 0.5]

    Compute the AUC-ROC score for each label's binary classifier:

    AUC-ROC for Label A: 0.667
    AUC-ROC for Label B: 0.833
    AUC-ROC for Label C: 0.667
    Calculate the macro-averaged AUC-ROC:

    Macro-Averaged AUC-ROC = (0.667 + 0.833 + 0.667) / 3 = 0.722

    paper=> Numbers reported are an unweighted mean across all 138 odor descriptors
    """
    y_test = dataset.y
    y_pred_probabilities = model.predict(dataset)
    # auc_roc_scores = []
    # for i in range(138):
    #     auc_roc_scores.append(roc_auc_score(y_test[:, i], y_pred_probabilities[:, i]))
    
    # macro_averaged_auc_roc = sum(auc_roc_scores) / len(auc_roc_scores)
    macro_averaged_auc_roc = roc_auc_score(y_test, y_pred_probabilities, average="macro")
    return macro_averaged_auc_roc
