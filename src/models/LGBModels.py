from src.models.ModelBaseClass import ModelBaseClass
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold

'''
[LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
'''


class LGBMSKLearnModel(ModelBaseClass):
    def __init__(self):
        pass
    
    def train(self, X_train, Y_train):
        X_train_train, X_train_dev, Y_train_train, Y_train_dev = train_test_split(X_train, Y_train, test_size=0.2)

        param_grid = {
            'num_leaves': [15, 31],  # Less complex models due to small dataset size
            'max_depth': [-1, 5, 10],  # -1 means no limit. Including shallow depths to prevent overfitting
            'learning_rate': [0.05, 0.1],  # Standard starting points for learning rate
            'n_estimators': [50, 100],  # Fewer trees to reduce computation time
            'min_child_samples': [20, 30],  # Minimum number of samples in a leaf
            'subsample': [0.8, 1.0],  # Standard subsampling rates
            'colsample_bytree': [0.8, 1.0],  # Standard feature subsampling rates
        }
        model = lgb.LGBMClassifier(verbose=-1)
        gbm = GridSearchCV(model, param_grid, cv=5)
        gbm.fit(X_train_train, Y_train_train, eval_set=[(X_train_dev, Y_train_dev)])

        self.model = gbm.best_estimator_
        return

    
    def predict(self, X):
        probabilities = self.model.predict(X)
        return (probabilities >= 0.5).astype(int)

    def report_trained_parameters(self):
        return  str(self.model)

    

class LGBMSKLearnModel2(ModelBaseClass):
    def __init__(self):
        self.lgb_estimator = lgb.LGBMClassifier(
            boosting_type='gbdt', 
            objective='binary', 
            num_boost_round=500, 
            learning_rate=0.01, 
            metric='balanced_accuracy',
            verbosity=-1,  # Suppressing warnings
            max_bin=255  # Default value, consider increasing if needed
        )
        self.model = None

    def train(self, X_train, Y_train):
        # Define the parameter grid
        param_grid = {
            'num_leaves': [15, 31],  # Simplified range
            'max_depth': [-1, 5],  # Reduced range
            'learning_rate': [0.05, 0.1],  # Moderate learning rates
            'n_estimators': [50, 100],  # Limited number of rounds
            'min_child_samples': [10, 20],  # Reduced range
            'min_sum_hessian_in_leaf': [1e-3, 1e-2],  # Adding this parameter
            'subsample': [0.8, 1.0],  # Standard subsampling
            'colsample_bytree': [0.8, 1.0],  # Feature subsampling
            'reg_alpha': [0.0, 0.1],  # L1 regularization
            'reg_lambda': [0.0, 0.1]  # L2 regularization
        }

        # Cross-validation setup
        gkf = KFold(n_splits=5, shuffle=True, random_state=42).split(X=X_train, y=Y_train)

        # Grid search
        gsearch = GridSearchCV(estimator=self.lgb_estimator, param_grid=param_grid, cv=gkf)
        self.model = gsearch.fit(X=X_train, y=Y_train).best_estimator_

        return
# class LGBMSKLearnModel2(ModelBaseClass):
#     def __init__(self):
#         self.lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', 
#                                                 num_boost_round=2000, learning_rate=0.01, 
#                                                 metric='balanced_accuracy')
#         self.model = None

#     def train(self, X_train, Y_train):
#         # Define the parameter grid
#         param_grid = {
#         'num_leaves': [15, 31, 50],  # Moderate complexity
#         'max_depth': [-1, 5, 10],  # Allowing some flexibility
#         'learning_rate': [0.05, 0.1],  # Moderate learning rates
#         'n_estimators': [50, 100, 200],  # Limited number of rounds for efficiency
#         'min_child_samples': [10, 20, 30],  # Adjusting for small dataset size
#         'subsample': [0.8, 1.0],  # Standard subsampling
#         'colsample_bytree': [0.8, 1.0],  # Feature subsampling
#         'reg_alpha': [0.0, 0.1],  # L1 regularization
#         'reg_lambda': [0.0, 0.1],  # L2 regularization
#         }

#         # Cross-validation setup
#         gkf = KFold(n_splits=5, shuffle=True, random_state=42).split(X=X_train, y=Y_train)

#         # Grid search
#         gsearch = GridSearchCV(estimator=self.lgb_estimator, param_grid=param_grid, cv=gkf)
#         # TODO: This method floods the command line with warnings. Suppress them or fix their underlying cause
#         self.model = gsearch.fit(X=X_train, y=Y_train).best_estimator_
#         return

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise Exception("Model has not been trained yet")

    def report_trained_parameters(self):
        if self.model is not None:
            return str(self.model)
        else:
            raise Exception("Model has not been trained yet")



class LGBMSKLearnModel3(ModelBaseClass):
    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        X_train_train, X_train_dev, Y_train_train, Y_train_dev = train_test_split(X_train, Y_train, test_size=0.2)

        lgb_train = lgb.Dataset(X_train_train, Y_train_train, params={'verbose': -1}, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_train_dev, Y_train_dev, params={'verbose': -1}, free_raw_data=False)
        self.model = lgb.train({'verbose': -1}, lgb_train, valid_sets=lgb_eval) # , verbose_eval=False
        return

    def predict(self, X):
        probabilities = self.model.predict(X)
        return (probabilities >= 0.5).astype(int)

    def report_trained_parameters(self):
        return  self.model.params