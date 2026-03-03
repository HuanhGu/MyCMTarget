from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_pred)

    return recall, precision, f1, accuracy, auc