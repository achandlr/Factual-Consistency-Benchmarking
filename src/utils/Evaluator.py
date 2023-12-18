from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import numpy as np
class Evaluator:
    @staticmethod
    def get_stats(y_true, y_pred):
        # Calculate the basic metrics using scikit-learn
        try:
            accuracy = accuracy_score(y_true, y_pred)
        except Exception as e:
            accuracy = np.nan
        try:
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        except Exception as e:
            balanced_accuracy = np.nan
        try:
            f1 = f1_score(y_true, y_pred)
        except Exception as e:
            f1 = np.nan
        try:
            sensitivity = recall_score(y_true, y_pred)  # Recall is the same as sensitivity
        except Exception as e:
            sensitivity = np.nan
        try:
            precision = precision_score(y_true, y_pred, zero_division=np.nan)
        except Exception as e:
            precision = np.nan
            
        try:
            # Calculating specificity requires TN and FP, which we get from the confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        except Exception as e:
            tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan
            specificity = np.nan

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
            'FN_count': fn,
            "y_pred" : y_pred,
            "y_true" : y_true
        }
        return results
    
    '''
    TODO: Ultamitely, either in this or the benchmark method. we want to have a dataframe that reports our benchmarks.
    The dataframe should be able to connect the following aspects.

    Model-Name, train_dataset,test_dataset stats or something like this
    '''
