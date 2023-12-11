from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from src.models.ModelBaseClass import ModelBaseClass
import pandas as pd
import numpy as np
import time
# General utilities from sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

# Ensemble methods
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

# Linear models
from sklearn.linear_model import LogisticRegression

# Decision Trees
from sklearn.tree import DecisionTreeClassifier

# Support Vector Machines
from sklearn.svm import SVC

# Naive Bayes classifiers
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Boosting frameworks outside of sklearn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Dummy classifier for baseline comparisons
from sklearn.dummy import DummyClassifier

class SKLearnModel(ModelBaseClass):
    def __init__(self, base_model, param_grid=None):
        """
        Initializes the SKLearnModel instance with a base model and an optional parameter grid for grid search.

        :param base_model: A scikit-learn model instance.
        :param param_grid: An optional dictionary defining the grid of parameters for grid search.
        """
        self.base_model = base_model
        self.param_grid = param_grid
        self.grid_search = None

    # def train(self, X_train, Y_train):
    #     if self.param_grid:
    #         self.grid_search = GridSearchCV(clone(self.base_model), self.param_grid, cv=5)
    #         self.grid_search.fit(X_train, Y_train)
    #     else:
    #         self.base_model.fit(X_train, Y_train)
    def train(self, X_train, Y_train):
        start_time = time.time()  # Start timing

        model_name = self.base_model.__class__.__name__  # Get the class name of the base model

        if self.param_grid:
            print(f"Starting GridSearchCV for {model_name}...")
            self.grid_search = GridSearchCV(clone(self.base_model), self.param_grid, cv=5)
            self.grid_search.fit(X_train, Y_train)
            grid_search_time = time.time() - start_time  # Time taken for grid search
            print(f"GridSearchCV for {model_name} completed. Time taken: {grid_search_time:.2f} seconds.")
        else:
            print(f"Starting training for {model_name}...")
            self.base_model.fit(X_train, Y_train)
            base_model_time = time.time() - start_time  # Time taken for base model training
            print(f"Training for {model_name} completed. Time taken: {base_model_time:.2f} seconds.")

        total_time = time.time() - start_time  # Total time taken for training
        print(f"Total training time for {model_name}: {total_time:.2f} seconds")

        # Output the timing information
        print(f"Total training time for {str(self.base_model)}: {total_time:.2f} seconds")
        if total_time > 0:  # Check to prevent division by zero
            if self.param_grid:
                print(f"GridSearchCV time: {grid_search_time:.2f} seconds ({grid_search_time / total_time * 100:.2f}%)")
            else:
                print(f"Base model training time: {base_model_time:.2f} seconds ({base_model_time / total_time * 100:.2f}%)")
        else:
            print("Training completed too quickly to measure.")

    # def train(self, X_train, Y_train):
    #     start_time = time.time()  # Start timing

    #     model_name = self.base_model.__class__.__name__  # Get the class name of the base model

    #     if self.param_grid:
    #         print(f"Starting GridSearchCV for {model_name}...")
    #         self.grid_search = GridSearchCV(clone(self.base_model), self.param_grid, cv=5)
    #         self.grid_search.fit(X_train, Y_train)
    #         grid_search_time = time.time() - start_time  # Time taken for grid search
    #         print(f"GridSearchCV for {model_name} completed. Time taken: {grid_search_time:.2f} seconds.")
    #     else:
    #         print(f"Starting training for {model_name}...")
    #         self.base_model.fit(X_train, Y_train)
    #         base_model_time = time.time() - start_time  # Time taken for base model training
    #         print(f"Training for {model_name} completed. Time taken: {base_model_time:.2f} seconds.")

    #     total_time = time.time() - start_time  # Total time taken for training
    #     print(f"Total training time for {model_name}: {total_time:.2f} seconds")


    #     # Output the timing information
    #     print(f"Total training time for {str(self.base_model)}: {total_time:.2f} seconds")
    #     if self.param_grid:
    #         print(f"GridSearchCV time: {grid_search_time:.2f} seconds ({grid_search_time / total_time * 100:.2f}%)")
    #     else:
    #         print(f"Base model training time: {base_model_time:.2f} seconds ({base_model_time / total_time * 100:.2f}%)")

    def predict(self, X):
        return self.grid_search.predict(X) if self.grid_search else self.base_model.predict(X)

    def report_trained_parameters(self):
        return self.grid_search.best_params_ if self.grid_search else {}


class RandomForestSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [None, 20, 40],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }
        # param_grid = {
        #     'n_estimators': [10, 50, 100, 200, 300, 400],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [None, 10, 20, 30, 40, 50],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'bootstrap': [True, False]
        # }
        super().__init__(RandomForestClassifier(), param_grid)

class GradientBoostingSKLearnModel(SKLearnModel):
    def __init__(self):
        # param_grid = {
        #     'n_estimators': [100, 200, 300, 400],
        #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
        #     'subsample': [0.5, 0.7, 1.0],
        #     'max_depth': [3, 4, 5, 6],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }

        param_grid = {
            'n_estimators': [100, 200],  # Reduced number of estimators
            'learning_rate': [0.1, 0.2],  # Simplified learning rates
            'subsample': [0.7, 1.0],  # Fewer subsampling options
            'max_depth': [3, 4],  # Lower max depth to reduce complexity
            'min_samples_split': [2],  # Only one option for simplicity
            'min_samples_leaf': [1]  # Only one option for simplicity
        }
        super().__init__(GradientBoostingClassifier(), param_grid)

class AdaBoostSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
        super().__init__(AdaBoostClassifier(), param_grid)



class DummySKLearnModel(SKLearnModel):
    def __init__(self):
        super().__init__(DummyClassifier(strategy='prior'), None)


class VotingSKLearnModel(SKLearnModel):
    def __init__(self, estimators):
        super().__init__(VotingClassifier(estimators=estimators, voting='hard'), None)

class LogisticRegressionSKLearnModel(SKLearnModel):
    def __init__(self):
        # param_grid = {
        #     'C': np.logspace(-4, 4, 20),
        #     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        #     'max_iter': [100, 200, 300, 400, 500]
        # }
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [1000, 2000, 3000]  # Increased max_iter
        }
        super().__init__(LogisticRegression(), param_grid)

class SVCSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'C': np.logspace(-2, 10, 13),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'] + list(np.logspace(-9, 3, 13))
        }
        super().__init__(SVC(), param_grid)

class DecisionTreeSKLearnModel(SKLearnModel):
    def __init__(self):
        # param_grid = {
        #     'criterion': ['gini', 'entropy'],
        #     'splitter': ['best', 'random'],
        #     'max_depth': [None] + list(range(1, 50)),
        #     'min_samples_split': range(2, 10),
        #     'min_samples_leaf': range(1, 10),
        #     'max_features': [None, 'auto', 'sqrt', 'log2']
        # }
        # param_grid = {
        #     'criterion': ['gini', 'entropy'],
        #     'splitter': ['best', 'random'],
        #     'max_depth': [None, 1, 2, 5, 10, 20, 30],  # Reduced and simplified range
        #     'min_samples_split': [2, 4, 6, 8],  # Fewer options
        #     'min_samples_leaf': [1, 3, 5, 7],  # Fewer options
        #     'max_features': [None, 'auto']  # Simplified options
        # }
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 1, 2, 5, 10, 20, 30],  # Simplified range
            'min_samples_split': [2, 4, 6, 8],         # Fewer options
            'min_samples_leaf': [1, 3, 5, 7],          # Fewer options
            'max_features': [None, 'sqrt', 'log2']     # Valid options
        }
        super().__init__(DecisionTreeClassifier(), param_grid)



class GaussianNBSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=100)
        }
        super().__init__(GaussianNB(), param_grid)

class MultinomialNBSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'alpha': np.linspace(0.0, 1.0, num=100)
        }
        super().__init__(MultinomialNB(), param_grid)

class BernoulliNBSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'alpha': np.linspace(0.0, 1.0, num=100)
        }
        super().__init__(BernoulliNB(), param_grid)

class KNeighborsSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_neighbors': range(1, 31),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
        super().__init__(KNeighborsClassifier(), param_grid)

class LDASKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': [None, 'auto'] + list(np.linspace(0.0, 1.0, num=50))
        }
        super().__init__(LinearDiscriminantAnalysis(), param_grid)


class XGBSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.7, 1.0]
        }
        super().__init__(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid)

class LGBMSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 70],
            'max_depth': [3, 4, 5, 6],
            'min_child_samples': [20, 30, 40],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.7, 1.0]
        }
        super().__init__(LGBMClassifier(), param_grid)

class CatBoostSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
        super().__init__(CatBoostClassifier(verbose=0), param_grid)


def instantiate_sk_learn_models():
    # Instantiate each model with its specific grid search parameters
    models = [
        # TODO: Uncomment the models that I have commented out. Commented out means that it works
        # RandomForestSKLearnModel(),
        GradientBoostingSKLearnModel(),
        AdaBoostSKLearnModel(),
        DummySKLearnModel(),
        LogisticRegressionSKLearnModel(),
        SVCSKLearnModel(),
        DecisionTreeSKLearnModel(),
        GaussianNBSKLearnModel(),
        MultinomialNBSKLearnModel(),
        BernoulliNBSKLearnModel(),
        KNeighborsSKLearnModel(),
        LDASKLearnModel(),
        XGBSKLearnModel(),
        LGBMSKLearnModel(),
        CatBoostSKLearnModel()
    ]

    # For VotingClassifier, we need to pass a list of model tuples
    # Here's an example of how you could do it
    # TODO: I am not sure if this is right or what it does
    voting_estimators = [
        ('rf', models[0].base_model),
        ('gb', models[1].base_model),
        ('ada', models[2].base_model)
    ]
    voting_model = VotingSKLearnModel(estimators=voting_estimators)
    models.append(voting_model)

    return models
