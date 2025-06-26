RNN Acceptor - Assignment
This README explains how to run the RNN acceptor model for binary sequence classification using LSTM, covering Part 1 (implementation and experimentation) and Part 2 (designing languages the model fails on).

Part 1: RNN Acceptor
Files
experiment.py: Trains and evaluates the LSTM-based acceptor.
gen_example.py: Generates datasets.
data/train.txt, data/test.txt: Training and testing datasets.

Requirements
Python 3.7+
PyTorch
matplotlib (optional for plotting)
Install dependencies with:

pip install torch matplotlib

Generating Data
Run the following to generate training and test data:

python gen_example.py

Running the Experiment
python experiment.py --train_file data/train.txt --test_file data/test.txt

To save a plot of training loss and accuracy per epoch:

python experiment.py --train_file data/train.txt --test_file data/test.txt --plot

Additional options:

--embedding_dim
--hidden_dim
--batch_size
--num_epochs
--learning_rate

Example

python experiment.py --train_file data/train.txt --test_file data/test.txt --num_epochs 20 --plot

