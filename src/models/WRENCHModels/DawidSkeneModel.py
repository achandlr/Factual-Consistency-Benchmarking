import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from src.models.ModelBaseClass import ModelBaseClass
import numpy as np
import logging
from abc import ABC, abstractmethod

# Losely based on https://github.com/kajyuuen/Dawid-skene/tree/master
class DawidSkeneModel(ModelBaseClass):
    def __init__(self, class_num=2, max_iter=100, tolerance=0.01):
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.error_rates = None
        self.predict_label = None
        self.dataset_tensor = None
        self.worker_num = None  # Initialize worker_num
        self.task_num = None    # Initialize task_num

    def _train(self, X_train, golden_labels):
        self.dataset_tensor = self._convert_to_tensor(X_train)
        self.worker_num = X_train.shape[1]  # Set the number of workers
        self.task_num = X_train.shape[0]    # Set the number of tasks

        self._run()

    def predict(self, X):
        # This method now handles new data
        new_data_tensor = self._convert_to_tensor(X)
        # TODO: changed from argmax to argmin
        return np.argmin(self._e_step(self.predict_label, self.error_rates, new_data_tensor), axis=1)

    def report_trained_parameters(self):
        return self.error_rates

    def _convert_to_tensor(self, X):
        num_tasks, num_workers = X.shape
        dataset_tensor = np.zeros((num_tasks, num_workers, self.class_num))
        for task_i in range(num_tasks):
            for worker_j in range(num_workers):
                label = int(X[task_i, worker_j])
                dataset_tensor[task_i, worker_j, label] = 1
        return dataset_tensor

    def _run(self):
        task_num, worker_num, _ = self.dataset_tensor.shape
        self.predict_label = np.random.rand(task_num, self.class_num)  # Random initialization
        self.predict_label = self.predict_label / self.predict_label.sum(axis=1, keepdims=True)

        iter_num = 0
        while iter_num < self.max_iter:
            self.error_rates = self._m_step(self.predict_label)
            self.predict_label = self._e_step(self.predict_label, self.error_rates, self.dataset_tensor)

            if iter_num % 10 == 0:
                logging.info(f"Iteration {iter_num}: Log-Likelihood = {self._get_likelihood()}")
            iter_num += 1

    def _m_step(self, predict_label):
        error_rates = np.zeros((self.worker_num, self.class_num, self.class_num))
        for i in range(self.class_num):
            worker_error_rate = np.dot(predict_label[:, i], self.dataset_tensor.transpose(1, 0, 2))
            sum_worker_error_rate = worker_error_rate.sum(1)
            error_rates[:, i, :] = np.divide(worker_error_rate, sum_worker_error_rate[:, None],
                                             out=np.zeros_like(worker_error_rate),
                                             where=sum_worker_error_rate[:, None] != 0)
        return error_rates

    def _e_step(self, predict_label, error_rates, data_tensor):
        task_num = data_tensor.shape[0]
        next_predict_label = np.zeros((task_num, self.class_num))
        for i in range(task_num):
            class_likelihood = self._get_class_likelihood(error_rates, data_tensor[i])
            next_predict_label[i] = predict_label.sum(0) / task_num * class_likelihood
            next_predict_label[i] /= next_predict_label[i].sum()
        return next_predict_label

    def _get_likelihood(self):
        log_L = 0
        for i in range(self.task_num):
            class_likelihood = self._get_class_likelihood(self.error_rates, self.dataset_tensor[i])
            log_L += np.log((self.predict_label.sum(0) / self.task_num * class_likelihood).sum())
        return log_L

    def _get_class_likelihood(self, error_rates, task_tensor):
        return np.power(error_rates.transpose(0, 2, 1), task_tensor[:, :, None]).prod(0).prod(1)

# # Example of using the class
# model = DawidSkeneModel()
# X_train = np.random.randint(0, 2, (100, 5))  # 100 tasks and 5 workers
# model.train(X_train)
# X_test = np.random.randint(0, 2, (20, 5))   # 20 new tasks for prediction
# predictions = model.predict(X_test)
# print("Predictions:", predictions)


# TODO: Numpy issues
'''
also might need conda install mkl-service
run into this issue: https://stackoverflow.com/questions/74893742/how-to-solve-attributeerror-module-numpy-has-no-attribute-bool
'''
# import pymc3 as pm

# class DavidSkeneModel(ABC):
#     def __init__(self):
#         self.model = None
#         self.trace = None

#     def train(self, X_train, Y_train):
#         # Number of items, annotators, and classes (binary)
#         I = X_train.shape[0]  # Number of items
#         J = X_train.shape[1]  # Number of annotators
#         K = 2  # Binary classification

#         # Convert X_train and Y_train to the format needed for Dawid-Skene
#         jj, ii, y = [], [], []
#         for i in range(I):
#             for j in range(J):
#                 jj.append(j)
#                 ii.append(i)
#                 y.append(int(X_train[i, j]))

#         # Convert to numpy arrays
#         jj = np.array(jj)
#         ii = np.array(ii)
#         y = np.array(y)

#         # Build the Dawid-Skene model
#         self.model = pm.Model()
#         with self.model:
#             # Priors
#             pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
#             theta = pm.Dirichlet('theta', a=np.ones((K, K)) + np.diag(np.ones(K)), shape=(J, K, K))

#             # Latent variable for true category of each item
#             z = pm.Categorical('z', p=pi, shape=I, testval=np.random.randint(0, K, I))

#             # Observations
#             y_obs = pm.Categorical('y_obs', p=theta[jj, z[ii]], observed=y)

#             # Sampling
#             step1 = pm.Metropolis(vars=[pi, theta])
#             step2 = pm.CategoricalGibbsMetropolis(vars=[z])
#             self.trace = pm.sample(5000, step=[step1, step2])

#     def predict(self, X):
#         # Predict based on the learned model
#         z_samples = self.trace['z'][-1000:, :]
#         z_hat = np.array([np.bincount(z_samples[:, i], minlength=2).argmax() for i in range(X.shape[0])])
#         return z_hat

#     def report_trained_parameters(self):
#         # Return model parameters
#         pi_estimates = np.mean(self.trace['pi'], axis=0)
#         theta_estimates = np.mean(self.trace['theta'], axis=0)
#         return {"class_prevalence": pi_estimates, "confusion_matrices": theta_estimates}
