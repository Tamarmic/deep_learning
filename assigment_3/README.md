
# RNN Acceptor - Assignment

This README explains how to run the RNN acceptor model for binary sequence classification using LSTM, covering Part 1 (implementation and experimentation) and Part 2 (designing languages the model fails on).

---

## Part 1: RNN Acceptor

### Files

- `experiment.py`: Trains and evaluates the LSTM-based acceptor.
- `gen_example1.py`: Generates datasets for languages 1–10.
- `data/langX_train.txt`, `data/langX_test.txt`: Training and testing datasets for each language (X = 1 to 10).
- `report1.docx`: Summary of experiments for Part 1.
- `training_plot.png`: Optional plot generated if `--plot` is used.

### Requirements

- Python 3.7+
- PyTorch
- matplotlib (optional for plotting)

Install dependencies with:
```bash
pip install torch matplotlib
```

### Generating Data

Run the following to generate training and test data for all 10 languages:
```bash
python gen_example1.py
```

### Running the Experiment

```bash
python experiment.py --train_file data/lang1_train.txt --test_file data/lang1_test.txt
```

To save a plot of training loss and accuracy per epoch:
```bash
python experiment.py --train_file data/lang1_train.txt --test_file data/lang1_test.txt --plot
```

Additional options:
- `--embedding_dim`
- `--hidden_dim`
- `--batch_size`
- `--num_epochs`
- `--learning_rate`

### Example

```bash
python experiment.py --train_file data/lang8_train.txt --test_file data/lang8_test.txt --num_epochs 20 --plot
```

---

## Part 2: Challenging the Acceptor

In Part 2, we explore the limitations of LSTM-based acceptors by constructing languages the model is expected to fail on. Languages 8–10 were designed to expose these weaknesses.

### Challenging Languages

- **Language 8**: Requires counting character frequencies and computing with ASCII values.
- **Language 9**: Requires comparing first and last characters (long-distance dependency).
- **Language 10**: Depends on a global property: string length parity (even vs odd).

### Files

- `gen_example1.py`: Includes generators for languages 8–10.
- `data/lang8_train.txt`, etc.: Generated data for each challenging case.
- `report2.docx`: Detailed failure analysis of these languages, with placeholders for performance plots.

### Expected Results

These languages are likely to confuse the RNN model due to:
- Arithmetic operations (Language 8)
- Long-distance dependencies (Language 9)
- Global string length features (Language 10)

Use the same training script as in Part 1 with:
```bash
python experiment.py --train_file data/lang8_train.txt --test_file data/lang8_test.txt --plot
```

---

## Contact

For any issues or clarification, refer to the assignment PDF or instructor.
