BiLSTM Tagger (Part 3)

This document describes how to use the files `bilstmTrain.py` and `bilstmPredict.py`
for training and evaluating a BiLSTM-based tagger on POS and NER datasets,
as required in Part 3 of the assignment.

---

Requirements
------------
- Python 3.8+
- PyTorch
- numpy
- matplotlib (for plotting)


---

Running the Tagger
------------------

Training:

    python bilstmTrain.py repr trainFile modelFile [options]

    Required arguments:
      repr         Representation mode: one of {a, b, c, d}
      trainFile    Path to training data (in CoNLL format)
      modelFile    Output file to save the trained model

    Optional arguments:
      --dev_file DEVFILE          Path to dev set (CoNLL format)
      --task TASK                Task type: pos or ner (used for labeling & metrics)
      --embedding_dim D          Word embedding dimension (default: 50)
      --hidden_dim D             BiLSTM hidden size (default: 100)
      --char_emb_dim D           Char embedding dim (used in b and d, default: 30)
      --char_hidden_dim D        Char-level LSTM hidden dim (default: 50)
      --epochs N                 Number of epochs (default: 5)
      --batch_size N             Batch size (default: 64)
      --lr LR                    Learning rate (default: 1e-3)

Example:

    python bilstmTrain.py a data/pos_train.txt pos_model.pt \
        --dev_file data/pos_dev.txt --task pos

---

Prediction:

    python bilstmPredict.py repr modelFile inputFile [--output_file path]

    Required arguments:
      repr         Representation mode: one of {a, b, c, d}
      modelFile    Trained model file (.pt)
      inputFile    Input CoNLL-style test file (unlabeled)

    Optional arguments:
      --output_file path         File to save the predictions (default: predictions.txt)

Example:

    python bilstmPredict.py a pos_model.pt data/pos_test.txt --output_file test4.pos

---

Output Format
-------------
- Prediction outputs are in CoNLL format:
      word1 TAG1
      word2 TAG2
      ...
    (Sentences separated by blank lines)

---

Evaluation & Reporting
----------------------

The training script saves dev-set accuracy every 500 sentences.
These are saved to files:
    dev_acc_{task}_{repr}.pkl

You can plot learning curves using the provided `plot_dev_accuracies.py`:

    python plot_dev_accuracies.py

It produces 2 plots:
  1. POS Dev Accuracy: 4 lines (a, b, c, d)
  2. NER Dev Accuracy: 4 lines (a, b, c, d)

Each curve shows accuracy (y-axis) vs. number of sentences seen / 100 (x-axis).

---



