BiLSTM Tagger README

This repository contains an implementation of a 2-layer BiLSTM sequence tagger supporting four different input representation modes:

Input Representation Modes

- a: Word embeddings only. Each word is represented by a learned embedding vector.
- b: Character-level LSTM embeddings only. Each word is represented by a character LSTM encoding over its characters.
- c: Embeddings plus subword representation (from Assignment 2). Placeholder implemented; replace with your own subword model.
- d: Concatenation of word embeddings (a) and character LSTM embeddings (b) followed by a linear projection.

Files

- bilstmTrain.py: Training script implementing all four representation modes.
- data_utils.py: Utility functions for dataset reading, vocabulary building, and encoding.
- bilstmPredict.py: (To be implemented) Script for predicting tags using a trained model.

Usage

Train the model with:

python bilstmTrain.py <repr> <train_file> <model_file> [--dev_file <dev_file>] [options]

- <repr>: One of a, b, c, or d, selecting the input representation mode.
- <train_file>: Path to the training data file.
- <model_file>: Path to save the trained model checkpoint.
- --dev_file: (Optional) Path to development data for evaluation during training.

Example:

python bilstmTrain.py a data/train.txt model_a.pt --dev_file data/dev.txt --epochs 10 --batch_size 32

Additional options

- --embedding_dim: Word/subword embedding dimension (default: 100)
- --hidden_dim: BiLSTM hidden layer size (default: 100)
- --char_emb_dim: Character embedding dimension (default: 30)
- --char_hidden_dim: Character LSTM hidden size (default: 50)
- --epochs: Number of training epochs (default: 5)
- --batch_size: Batch size (default: 32)
- --lr: Learning rate (default: 0.001)

Data format

Input files should be in the CoNLL style format:

word tag
word tag

Sentences are separated by blank lines.

Notes

- For modes b and d, character-level representations require building a character vocabulary from the training data.
- Mode c currently uses a placeholder embedding. Replace with your assignment 2 subword embedding logic.
- The model saves a checkpoint containing the trained parameters and vocabularies (word2idx, tag2idx, char2idx).
- The code handles padding and masking automatically for variable-length sequences.


Enjoy training your BiLSTM tagger!