import torch
import numpy as np
from train import train
from torch import nn
from model import LSTMModel, init_weights
from functions import plot, evaluate, load_X, load_y
from datetime import datetime
import config as cfg
import data as df
import sys
import os

# Data file to load X and y values
y_test_path = df.y_test_path
y_train_path = df.y_train_path
X_test_signals_paths = df.X_test_signals_paths
X_train_signals_paths = df.X_train_signals_paths

# LSTM Neural Network's internal structure
diag = cfg.diag
epochs = cfg.n_epochs
n_hidden = cfg.n_hidden
clip_val = cfg.clip_val
n_classes = cfg.n_classes
weight_decay = cfg.weight_decay
learning_rate = cfg.learning_rate

# Training on gpu
if (torch.cuda.is_available() ):
    print('Training on GPU')
    train_on_gpu = torch.cuda.is_available()
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')

# Create a 'results' directory if it doesn't exist
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def main():
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)
    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # Input Data
    training_data_count = len(X_train)
    test_data_count = len(X_test)
    n_steps = len(X_train[0])
    n_input = len(X_train[0][0])

    # Info
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    for lr in learning_rate:
        arch = cfg.arch
        if arch['name'] == 'LSTM1' or arch['name'] == 'LSTM2':
            net = LSTMModel()
        else:
            print("Incorrect architecture chosen. Please check architecture given in config.py. Program will exit now! :( ")
            sys.exit()
        net.apply(init_weights)
        print(diag)

        # Hyperparam
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        net = net.float()

        # Create a directory for the current run with timestamp
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(results_dir, f'run_{run_timestamp}')
        os.makedirs(run_dir, exist_ok=True)
        models_dir = os.path.join(run_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Start training and eval
        params = train(net, X_train, y_train, X_test, y_test, opt=opt, criterion=criterion, epochs=epochs, clip_val=clip_val)
        evaluate(params['best_model'], X_test, y_test, criterion, run_dir)

        # Save the best and last models
        torch.save(params['best_model'].state_dict(), os.path.join(models_dir, 'best.pth'))
        torch.save(net.state_dict(), os.path.join(models_dir, 'last.pth'))

        # Save the plots
        plot(params['epochs'], params['train_loss'], params['test_loss'], 'loss', lr, run_dir)
        plot(params['epochs'], params['train_accuracy'], params['test_accuracy'], 'accuracy', lr, run_dir)

main()