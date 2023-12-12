from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import numpy as np
class Evaluator:
    @staticmethod
    def get_stats(y_true, y_pred):
        # Calculate the basic metrics using scikit-learn
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)  # Recall is the same as sensitivity
        precision = precision_score(y_true, y_pred, zero_division=np.nan)
        
        # Calculating specificity requires TN and FP, which we get from the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        # Compile the results into a dictionary
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'TP_count': tp,
            'TN_count': tn,
            'FP_count': fp,
            'FN_count': fn
        }
        return results
    
    '''
    TODO: Ultamitely, either in this or the benchmark method. we want to have a dataframe that reports our benchmarks.
    The dataframe should be able to connect the following aspects.

    Model-Name, train_dataset,test_dataset stats or something like this
    '''
