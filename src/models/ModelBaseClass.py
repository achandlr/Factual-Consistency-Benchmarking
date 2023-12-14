from abc import ABC, abstractmethod
from src.utils.logger import setup_logger
import time
class ModelBaseClass(ABC):

    def train_with_timing_stats(self, X_train, Y_train):
        """
        This method should be used to train the model and also collect timing stats.
        """
        # TODO: If necessary, potentially store logger as part of class and training times better
        logger = setup_logger()
        start_time = time.time()
        self.train(X_train, Y_train)
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time/60} minutes \n\n")


    @abstractmethod
    def train(self, X_train, Y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def report_trained_parameters(self, X):
        # TODO: This method should report the parameter weights
        pass


    # @abstractmethod
    # def tune(self, X_train, Y_train, param_grid):
    #     pass



# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.linear_model import LogisticRegression
# from mlens.ensemble import SuperLearner
# import numpy as np

# class Models:
#     def __init__(self, base_classifiers, param_grids):
#         self.base_classifiers = base_classifiers
#         self.param_grids = param_grids
#         self.ensemble = SuperLearner(scorer=accuracy_score, random_state=42, verbose=2)
#         self.setup_ensemble()

#     def setup_ensemble(self):
#         for name, clf in self.base_classifiers.items():
#             self.ensemble.add(clf)
#         self.ensemble.add_meta(LogisticRegression())



#     def tune_and_evaluate(self, X_train, Y_train, X_test, Y_test):
#         complete_param_grid = self.format_param_grid()
#         grid_search = GridSearchCV(self.ensemble, param_grid=complete_param_grid, cv=5, scoring=make_scorer(accuracy_score))
#         grid_search.fit(X_train, Y_train)

#         print("Best parameters:", grid_search.best_params_)
#         print("Best score:", grid_search.best_score_)

#         y_pred = grid_search.predict(X_test)
#         return y_pred, grid_search.best_params_, grid_search.best_score_

#     def format_param_grid(self):
#         # [Your code to format the complete_param_grid goes here]
#         return complete_param_grid



# '''
# Needed Subclasses:

# Majority Voting
# Snorkel
# MeTal
# Weasel
# Consensus. this is an issue because we do not have code access to this so rather we will have it empty for now raise NotImplemented
# methods - 

# '''