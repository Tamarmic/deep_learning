# 🧠 Assignment 2 – Neural Window-Based Tagger & Char-Level Language Model

This project implements a neural **window-based tagger** for Part-of-Speech (POS) and Named Entity Recognition (NER), along with several enhancements and a separate **character-level language model**.

---

## 📦 Installation Requirements

This project requires the following Python packages:

-   Python
-   PyTorch
-   NumPy
-   Matplotlib

To install them:

```bash
pip install torch numpy matplotlib
```

---

## 🧪 Part 1 – Basic Window-Based Tagger (Random Embeddings)

Trains a window-based feedforward neural tagger using **randomly initialized** word embeddings.

### 🏃‍♂️ Run:

```bash
python tagger1.py --task pos --part 1
```

or for NER:

```bash
python tagger1.py --task ner --part 1
```

---

## 🧠 Part 2 – Word Similarity with Pretrained Embeddings

Computes top-K most similar words using cosine similarity on pretrained embeddings.

### 🏃‍♂️ Run:

```bash
python top_k.py
```

### 🔧 Options:

-   `--vocab_file` Path to vocab (default: `embeddings/vocab.txt`)
-   `--vectors_file` Path to pretrained vectors (default: `embeddings/wordVectors.txt`)
-   `--k` Number of top similar words to return (default: 5)

📤 Output is printed to the console for copy-pasting into `part2.pdf`.

---

## 💬 Part 3 – Tagger with Pretrained Word Embeddings

Uses pretrained word embeddings instead of random initialization for word vectors.

### 🏃‍♂️ Run:

```bash
python tagger1.py --task pos --part 3 --use_pretrained_embeddings
```

### 📂 Required Files:

-   `embeddings/vocab.txt`
-   `embeddings/wordVectors.txt`

---

## 🧩 Part 4 – Tagger with Subword Prefix + Suffix Features

Adds prefix and suffix embeddings per word to improve tagging robustness.

### 🏃‍♂️ Subwords + Random Embeddings:

```bash
python tagger1.py --task pos --part 4 --use_subwords
```

### 🧠 Subwords + Pretrained Embeddings:

```bash
python tagger1.py --task ner --part 4 --use_subwords --use_pretrained_embeddings
```

---

## 🧬 Part 5 – CNN Subword Tagger (Ma & Hovy Style)

Extracts subword features using a CNN over character sequences for each word.

### 🏃‍♂️ CNN + Random Embeddings:

```bash
python tagger1.py --task pos --part 5 --use_cnn_subwords
```

### 🧠 CNN + Pretrained Embeddings:

```bash
python tagger1.py --task ner --part 5 --use_cnn_subwords --use_pretrained_embeddings
```

⚠️ **Important**:  
You **cannot use** both `--use_subwords` and `--use_cnn_subwords` simultaneously.

---

## ✍️ Part 6 – Character-Level N-gram Language Model

Trains a character-level n-gram language model on a plain text file (e.g., `eng.txt`).  
Predicts the next character from the previous `k` characters using an MLP.

### 🏃‍♂️ Run:

```bash
python char_lm.py --file eng.txt --k 5 --sample_prefix "the "
```

### 🔧 Options:

-   `--file` Path to training text file (default: `eng.txt`)
-   `--k` Context length (default: 5)
-   `--emb_dim` Character embedding dim (default: 30)
-   `--hidden_dim` Hidden layer size (default: 100)
-   `--num_epochs` Number of epochs (default: 10)
-   `--batch_size` Batch size (default: 128)
-   `--learning_rate` Learning rate (default: 0.001)
-   `--sample_every` Interval to generate samples (default: 2)
-   `--sample_prefix` Prompt used when generating text
-   `--sample_length` Number of characters to sample (default: 100)

📤 Outputs:

-   Training loss curve in `charlm_outputs/loss_char_lm.png`
-   Saved model in `charlm_outputs/char_lm.pt`

---

## ⚙️ Global Arguments (For `tagger1.py`)

```bash
--task pos|ner               # Required task
--part 1|3|4|5               # Required part
--embedding_dim 50          # Word embedding size
--hidden_dim 100            # Hidden layer size
--batch_size 32             # Batch size
--num_epochs 10             # Number of training epochs
--learning_rate 0.001       # Learning rate
--weight_decay 1e-5         # L2 regularization
--window_size 2             # Context window on each side
--use_pretrained_embeddings # Use pretrained word vectors
--use_subwords              # Enable prefix+suffix embeddings
--use_cnn_subwords          # Enable CNN over characters (Part 5)
```

---

## 📁 Output Files

After each run, the following folders are populated:

-   `saved_models/` – PyTorch model checkpoints  
    → `model_{task}_{suffix}.pt`

-   `logs/` – Training logs  
    → `loss_{task}_{suffix}_train.npy`, `acc_{task}_{suffix}_dev.npy`

-   `plots/` – Training/validation curves  
    → `loss_{task}_{suffix}.png`, `acc_{task}_{suffix}.png`

-   `predictions/` – Tag predictions for test set  
    → `test_{task}_{suffix}.txt` (e.g., `test_ner_p5+pre+cnn.txt`)

The `{suffix}` is auto-generated using:

-   `p1`, `p3`, `p4`, `p5` = part number
-   `+pre` = pretrained embeddings
-   `+sub` = subword prefixes/suffixes
-   `+cnn` = CNN subwords

---

## ✅ Notes

-   All data should be placed under folders like `pos/train`, `pos/dev`, `pos/test`, `ner/train`, etc.
-   Embeddings should be formatted to match `vocab.txt` and `wordVectors.txt`.
-   Dev accuracy is computed during training and saved in logs.

---
