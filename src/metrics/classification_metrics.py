from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred, average="weighted"):
        return precision_score(y_true, y_pred, average=average)

    @staticmethod
    def recall(y_true, y_pred, average="weighted"):
        return recall_score(y_true, y_pred, average=average)

    @staticmethod
    def f1(y_true, y_pred, average="weighted"):
        return f1_score(y_true, y_pred, average=average)

    @staticmethod
    def roc_auc(y_true, y_pred_proba, multi_class="ovr"):
        return roc_auc_score(y_true, y_pred_proba, multi_class=multi_class)