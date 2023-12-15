import numpy as np
import math
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, recall_score

from src.models.ModelBaseClass import ModelBaseClass

class ConditionalLR(ModelBaseClass):
    def __init__(self):
        pass

    @staticmethod
    def ensemble_prediction(input_array, function_to_TPFP_reward, function_to_TNFN_reward):
        pred = 0
        for idx, label in enumerate(input_array):
            if label:
                pred += function_to_TPFP_reward[idx]
            else:
                pred -= function_to_TNFN_reward[idx]
        return pred > 0
    
    @staticmethod
    def convert_ratio_to_reward(ratio, base):
        if ratio <= 0:
            raise ValueError("Ratio must be positive")
        return max(0, math.log2(ratio) * base)

    @staticmethod
    def calculate_ratios(X, Y):
        ratios = {'TPFP': {}, 'TNFN': {}}
        Y = np.array(Y).astype(int)

        for i in range(X.shape[1]):
            y_pred = np.array(X[:, i]).astype(int)
            try:
                tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()
            except ValueError as e:
                print(f"Error in function {i}: {e}")
                continue

            ratios['TPFP'][i] = max(1, tp / max(1, fp))
            ratios['TNFN'][i] = max(1, tn / max(1, fn))

        return ratios

    @staticmethod
    def optimize_rewards(X, Y, ratios):
        best_accuracy = 0
        best_base_TPFP = best_base_TNFN = 1
        for base_TPFP in np.linspace(0.1, 10, 20):
            for base_TNFN in np.linspace(0.1, 10, 50):
                rewards = {
                    'TPFP': {k: ConditionalLR.convert_ratio_to_reward(v, base_TPFP) for k, v in ratios['TPFP'].items()},
                    'TNFN': {k: ConditionalLR.convert_ratio_to_reward(v, base_TNFN) for k, v in ratios['TNFN'].items()}
                }
                predictions = [ConditionalLR.ensemble_prediction(row, rewards['TPFP'], rewards['TNFN']) for row in X]
                accuracy = balanced_accuracy_score(Y, predictions)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_base_TPFP = base_TPFP
                    best_base_TNFN = base_TNFN

        return best_base_TPFP, best_base_TNFN

    def _train(self, X_train, Y_train):
        ratios = ConditionalLR.calculate_ratios(X_train, Y_train)
        best_base_TPFP, best_base_TNFN = ConditionalLR.optimize_rewards(X_train, Y_train, ratios)
        self.optimized_rewards = {
            'TPFP': {k: ConditionalLR.convert_ratio_to_reward(v, best_base_TPFP) for k, v in ratios['TPFP'].items()},
            'TNFN': {k: ConditionalLR.convert_ratio_to_reward(v, best_base_TNFN) for k, v in ratios['TNFN'].items()}
        }
        predictions_train = [ConditionalLR.ensemble_prediction(row, self.optimized_rewards['TPFP'], self.optimized_rewards['TNFN'] ) for row in X_train]
        return

    def predict(self, X):
        predictions_test = [ConditionalLR.ensemble_prediction(row, self.optimized_rewards['TPFP'], self.optimized_rewards['TNFN']) for row in X]
        return predictions_test

    def report_trained_parameters(self):
        return self.optimized_rewards

