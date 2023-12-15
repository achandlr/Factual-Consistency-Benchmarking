import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod
import time
import logging
from src.models.ModelBaseClass import ModelBaseClass
from itertools import product
import random

# Note: This function was modified so that y_test is not used
def prepare_data(X_train, X_test, Y_train, batch_size=64):
    # Convert data to PyTorch tensors, ensuring they are of numeric type
    X_train_tensor = torch.tensor(X_train.astype(np.float32)) if X_train is not None else None
    Y_train_tensor = torch.tensor(Y_train.astype(np.float32)) if Y_train is not None else None
    X_test_tensor = torch.tensor(X_test.astype(np.float32)) if X_test is not None else None

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor) if X_train_tensor is not None and Y_train_tensor is not None else None
    test_dataset = TensorDataset(X_test_tensor) if X_test_tensor is not None else None

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset is not None else None

    return train_loader, test_loader

def train_model(model, train_loader, epochs=100, learning_rate=0.001, l2_regularization=None, optimizer_type='Adam',show_loss=False):
    # Selecting the optimizer based on the optimizer_type
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization if l2_regularization is not None else 0)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_regularization if l2_regularization is not None else 0)
    else:
        raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")

    # TODO: Make criterian take in weights that correspond to inverse frequency of binary labels or something along these lines
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
        if show_loss:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def test_model(model, test_loader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            output = model(data[0])
            predicted = (output.squeeze() > 0.5).float() # Convert to binary predictions
            y_pred.extend(predicted.tolist())

    # Convert to numpy array for return
    y_pred = np.array(y_pred)

    return y_pred

class NeuralModel2Layer(ModelBaseClass, nn.Module):
    def __init__(self, intermediate_size=50, batch_size=64, epochs=100, learning_rate=0.001, dropout_rate=None, l2_regularization=None, use_batch_norm=False, optimizer_type='Adam'):
        super(NeuralModel2Layer, self).__init__()
        # Store parameters, but don't initialize layers yet
        self.intermediate_size = intermediate_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        self.use_batch_norm = use_batch_norm
        self.optimizer_type = optimizer_type

    def initialize_layers(self, input_size):
        # Initialize layers now that input_size is known
        self.fc1 = nn.Linear(input_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate is not None else None
        self.batch_norm = nn.BatchNorm1d(self.intermediate_size) if self.use_batch_norm else None

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def _train(self, X_train, Y_train):
        # Determine input size from X_train and initialize layers
        input_size = X_train.shape[1]
        self.initialize_layers(input_size)

        train_loader, _ = prepare_data(X_train, None, Y_train, batch_size=self.batch_size)
        train_model(self, train_loader, epochs=self.epochs, learning_rate=self.learning_rate, l2_regularization=self.l2_regularization, optimizer_type=self.optimizer_type)

    def predict(self, X):
        # Correctly passing X as the test data and ignoring the train_loader
        _, test_loader = prepare_data(None, X, None, batch_size = self.batch_size)
        return test_model(self, test_loader)

    def report_trained_parameters(self):
        trained_paramaters = ""
        for name, param in self.named_parameters():
            if param.requires_grad:
                trained_paramaters += f"{name}: {param.data}\n"
        return trained_paramaters


def load_pytorch_models():
    pytorch_models = []
    baseline_params = {
        'intermediate_size': 32, 'batch_size': 16, 'epochs': 20,
        'learning_rate': 0.001, 'dropout_rate': 0.2, 'l2_regularization': 0.01,
        'use_batch_norm': True, 'optimizer_type': 'Adam'
    }

    # Define parameter groups as a dictionary
    groups = {
        'intermediate_size': [8, 16, 32, 64],
        'batch_size': [4, 8, 16, 32],
        'epochs': [5, 20, 100],
        'learning_rate': [0.001, 0.01],
        'dropout_rate': [None, 0.1, 0.2, 0.4],
        'l2_regularization': [0.01, 0.001],
        'use_batch_norm': [True, False],
        'optimizer_type': ['Adam', 'SGD']
    }

    # Create model variations
    for param, values in groups.items():
        for value in values:
            new_params = baseline_params.copy()
            new_params[param] = value
            model = NeuralModel2Layer(**new_params)
            pytorch_models.append(model)

    # Assert the number of models
    assert len(pytorch_models) == sum(len(values) for values in groups.values())

    return pytorch_models
