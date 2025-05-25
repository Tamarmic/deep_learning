# Part 1: LSTM Acceptor for Formal Language Classification

This part implements a neural sequence acceptor using nn.LSTMCell (not nn.LSTM) to recognize sequences matching a formal regular pattern.

# ✅ Pattern to Learn:

The network is trained to distinguish strings of the form:

[1-9]+ a+ [1-9]+ b+ [1-9]+ c+ [1-9]+ d+ [1-9]+

from incorrect variants.

## 🔧 Requirements

Python 3

PyTorch

Matplotlib (for plots)

Install dependencies:

pip instarch matplotlibll to

## 📂 File Overview

experiment.py: main training and evaluation code for the LSTMCell model

gen_examples.py: generates synthetic data for training and testing

# 📊 Run Instructions

## Step 1: Generate data

python gen_examples.py

This will create data/train.txt and data/test.txt with 5000 positive and 5000 negative examples each.

## Step 2: Train the model

python experiment.py --train_file data/train.txt --test_file data/test.txt --plot

## Optional arguments:

* --embedding_dim (default=16): dimension of character embeddings

* --hidden_dim (default=128): hidden size of LSTMCell

* --batch_size (default=64)

* --num_epochs (default=10)

* --learning_rate (default=0.001)

* --plot: if set, saves PNG plots of loss and accuracy curves

# 📈 Outputs

* Console logs showing epoch-by-epoch loss and accuracy

* Final test accuracy and total training time

* If --plot is used:

* * plot_loss.png: training loss vs. epoch

* * plot_acc.png: test accuracy vs. epoch
