# Assignment 2 – Window-Based Sequence Tagger

This project implements a neural window-based tagger for POS and NER, with extensions including:

-   Pretrained word embeddings
-   Subword features (prefixes + suffixes)
-   Word similarity via pretrained vectors

---

## 📦 Requirements

-   Python 3
-   PyTorch
-   NumPy
-   Matplotlib

Install missing packages:

```
pip install torch numpy matplotlib
```

---

## 🧪 Part 1: Window-Based Tagger (Random Embeddings)

Train and evaluate a simple window-based tagger using random word embeddings.

### POS Tagging:

```
python tagger1.py --task pos --part 1
```

### NER Tagging:

```
python tagger1.py --task ner --part 1
```

---

## 🧠 Part 2: Most Similar Words Using Pretrained Embeddings

Find top-k similar words based on cosine similarity using pretrained embeddings:

```
python top_k.py
```

### Arguments:

-   `--vocab_file` (default: `embeddings/vocab.txt`)
-   `--vectors_file` (default: `embeddings/wordVectors.txt`)
-   `--k` (default: 5)

**Output**: Console printout of top-k similar words for test words (used in `part2.pdf`).

---

## 💬 Part 3: Tagger with Pretrained Word Embeddings

Use pretrained word embeddings instead of random ones:

```
python tagger1.py --task pos --part 3 --use_pretrained_embeddings
```

-   Required files:
    -   `embeddings/vocab.txt`
    -   `embeddings/wordVectors.txt`

---

## 🧩 Part 4: Tagger with Subword Features (Prefix + Suffix)

Add prefix and suffix embeddings to improve tagging, optionally combined with pretrained embeddings.

### Subwords + Random Embeddings:

```
python tagger1.py --task pos --part 4 --use_subwords
```

### Subwords + Pretrained Embeddings:

```
python tagger1.py --task ner --part 4 --use_subwords --use_pretrained_embeddings
```

---

## ⚙️ Common Arguments

You can override any of the following (defaults shown):

```
--task pos
--part 1
--embedding_dim 50
--hidden_dim 100
--batch_size 32
--num_epochs 10
--learning_rate 0.001
--window_size 2
--use_pretrained_embeddings
--use_subwords
```

---

## 📁 Outputs

After running, these folders will be populated:

-   `saved_models/`: Trained PyTorch model files  
    → `model{suffix}_{task}.pt`

-   `logs/`: Training loss and dev accuracy logs  
    → `loss{suffix}_{task}_train.npy`, `acc{suffix}_{task}_dev.npy`

-   `plots/`: PNG graphs of loss and accuracy  
    → `loss{suffix}_{task}.png`, `acc{suffix}_{task}.png`

-   `predictions/`: Predicted tags on test set  
    → `test{suffix}.{task}` (e.g., `test4.pos`)

Use suffixes:

-   `1` → Part 1
-   `3` → Part 3
-   `4` → Part 4
