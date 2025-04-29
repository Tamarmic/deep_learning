import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import data_utils
from models import WindowTagger


# Helper Functions
def create_windows(word_indices, window_size=2):
    windows = []
    for i in range(window_size, len(word_indices) - window_size):
        window = word_indices[i - window_size : i + window_size + 1]
        windows.append(window)
    return windows


def evaluate_model(
    model, data, word2idx, tag2idx, idx2tag, task_type="pos", window_size=2
):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for sentence in data:
            word_idxs, tag_idxs = data_utils.encode_sentence(
                sentence, word2idx, tag2idx
            )
            word_idxs_padded = data_utils.pad_sentence(word_idxs, window_size)
            windows = create_windows(word_idxs_padded, window_size)

            X = torch.tensor(windows)
            logits = model(X)
            predictions = torch.argmax(logits, dim=1)

            for pred, gold in zip(predictions, tag_idxs):
                gold_tag = idx2tag[gold]
                pred_tag = idx2tag[pred.item()]
                if task_type == "pos" or not (gold_tag == "O" and pred_tag == "O"):
                    correct += int(pred == gold)
                    total += 1

    return correct / total if total > 0 else 0.0


def train_model(
    model,
    train_loader,
    dev_data,
    word2idx,
    tag2idx,
    idx2tag,
    optimizer,
    loss_fn,
    num_epochs,
    task_type,
    window_size,
    device,
):
    train_losses = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        dev_acc = evaluate_model(
            model, dev_data, word2idx, tag2idx, idx2tag, task_type, window_size
        )
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Dev Accuracy: {dev_acc:.4f}"
        )
        train_losses.append(epoch_loss)
        dev_accuracies.append(dev_acc)

    return train_losses, dev_accuracies


def predict_test(
    model, data, word2idx, idx2tag, output_file, window_size=2, device="cpu"
):
    model.eval()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as fout:
        with torch.no_grad():
            for sentence in data:
                word_idxs = [word2idx.get(w, word2idx["<UNK>"]) for w in sentence]
                padded = [0] * window_size + word_idxs + [0] * window_size
                windows = [
                    padded[i - window_size : i + window_size + 1]
                    for i in range(window_size, len(padded) - window_size)
                ]
                X = torch.tensor(windows).to(device)
                preds = torch.argmax(model(X), dim=1)
                for word, pred in zip(sentence, preds):
                    fout.write(f"{word} {idx2tag[pred.item()]}\n")
                fout.write("\n")


def plot_and_save(y_values, title, ylabel, filename):
    plt.figure()
    plt.plot(y_values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["pos", "ner"], default="pos")
    parser.add_argument(
        "--use_pretrained_embeddings",
        action="store_true",
        help="Use pre-trained word embeddings instead of random initialization",
    )
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--window_size", type=int, default=2)
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    # Load data
    train_data = data_utils.read_dataset(f"{args.task}/train")
    dev_data = data_utils.read_dataset(f"{args.task}/dev")
    test_data = data_utils.read_dataset(f"{args.task}/test", labeled=False)

    # Prepare vocab
    word2idx, tag2idx, idx2tag = data_utils.build_vocab(train_data)

    # Prepare embeddings
    if args.use_pretrained_embeddings:
        print("Loading pre-trained embeddings...")
        pretrain_word2vec = data_utils.load_embeddings(
            "embeddings/vocab.txt", "embeddings/wordVectors.txt"
        )

        scale = np.sqrt(1.0 / args.embedding_dim)
        embedding_matrix = np.random.uniform(
            -scale, scale, (len(word2idx), args.embedding_dim)
        )

        for word, idx in word2idx.items():
            if word in pretrain_word2vec:
                embedding_matrix[idx] = pretrain_word2vec[word]

        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    else:
        embedding_matrix = None

    # Prepare model
    model = WindowTagger(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(tag2idx),
        window_size=args.window_size,
        pretrained_embedding=embedding_matrix,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # Prepare Dataloader
    all_windows, all_tags = [], []
    for sentence in train_data:
        word_idxs, tag_idxs = data_utils.encode_sentence(sentence, word2idx, tag2idx)
        word_idxs_padded = data_utils.pad_sentence(word_idxs, args.window_size)
        windows = create_windows(word_idxs_padded, args.window_size)
        all_windows.extend(windows)
        all_tags.extend(tag_idxs)

    train_dataset = TensorDataset(torch.tensor(all_windows), torch.tensor(all_tags))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Train
    train_losses, dev_accuracies = train_model(
        model,
        train_loader,
        dev_data,
        word2idx,
        tag2idx,
        idx2tag,
        optimizer,
        loss_fn,
        args.num_epochs,
        args.task,
        args.window_size,
        device,
    )

    file_suffix = "3" if args.use_pretrained_embeddings else "1"

    # Save model and logs
    torch.save(model.state_dict(), f"saved_models/model{file_suffix}_{args.task}.pt")
    np.save(f"logs/loss{file_suffix}_{args.task}_train.npy", np.array(train_losses))
    np.save(f"logs/acc{file_suffix}_{args.task}_dev.npy", np.array(dev_accuracies))

    # Plot
    plot_and_save(
        train_losses,
        f"Training Loss ({args.task})",
        "Loss",
        f"plots/loss{file_suffix}_{args.task}.png",
    )
    plot_and_save(
        dev_accuracies,
        f"Dev Accuracy ({args.task})",
        "Accuracy",
        f"plots/acc{file_suffix}_{args.task}.png",
    )

    # Predict
    output_file = f"predictions/test{file_suffix}.{args.task}"
    predict_test(
        model, test_data, word2idx, idx2tag, output_file, args.window_size, device
    )

    print(f"Finished training and prediction for {args.task.upper()}!")
