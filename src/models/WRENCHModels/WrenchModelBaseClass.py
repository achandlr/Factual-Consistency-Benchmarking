import subprocess
from abc import ABC, abstractmethod
from src.models.ModelBaseClass import ModelBaseClass

class WrenchModelBaseClass(ModelBaseClass, ABC):
    def __init__(self, model_name, env_path):
        self.model_name = model_name
        self.env_path = env_path

    def _run_subprocess(self, command):
        # Activate the virtual environment and run the command
        activate_env = f'source {self.env_path}/bin/activate'
        subprocess.run(f'{activate_env} && {command}', shell=True, executable='/bin/bash')

    def train(self, X_train, Y_train):
        # Serialize X_train, Y_train, save to file, and run the training in subprocess
        pass

    def predict(self, X):
        # Run prediction in subprocess and return Y_pred
        pass

    def report_trained_parameters(self):
        # Retrieve and return model parameters from the subprocess
        pass