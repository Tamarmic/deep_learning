import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import data_utils
from models import WindowTagger, WindowTaggerSubword, WindowTaggerCNNSubword


def create_windows(word_indices, window_size=2):
    windows = []
    for i in range(window_size, len(word_indices) - window_size):
        window = word_indices[i - window_size : i + window_size + 1]
        windows.append(window)
    return windows


def plot_and_save(y_values, title, ylabel, filename):
    plt.figure()
    plt.plot(y_values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


def evaluate_model(
    model,
    data,
    word2idx,
    tag2idx,
    idx2tag,
    task_type="pos",
    window_size=2,
    prefix2idx=None,
    suffix2idx=None,
    use_subwords=False,
    device="cpu",
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

            X = torch.tensor(windows).to(device)
            if use_subwords:
                pre_idxs, suf_idxs = data_utils.encode_prefix_suffix(
                    sentence, prefix2idx, suffix2idx
                )
                pre_windows = create_windows(
                    data_utils.pad_sentence(pre_idxs, window_size), window_size
                )
                suf_windows = create_windows(
                    data_utils.pad_sentence(suf_idxs, window_size), window_size
                )
                prefix_tensor = torch.tensor(pre_windows).to(device)
                suffix_tensor = torch.tensor(suf_windows).to(device)
                logits = model(X, prefix_tensor, suffix_tensor)
            else:
                logits = model(X)
            predictions = torch.argmax(logits, dim=1)

            for pred, gold in zip(predictions, tag_idxs):
                gold_tag = idx2tag[gold]
                pred_tag = idx2tag[pred.item()]
                if task_type == "pos" or not (gold_tag == "O" and pred_tag == "O"):
                    correct += int(pred == gold)
                    total += 1

    return correct / total if total > 0 else 0.0


def predict_test(
    model,
    data,
    word2idx,
    idx2tag,
    output_file,
    window_size=2,
    prefix2idx=None,
    suffix2idx=None,
    use_subwords=False,
    device="cpu",
):
    model.eval()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as fout:
        with torch.no_grad():
            for sentence in data:
                word_idxs = [
                    word2idx.get(w, word2idx[data_utils.UNK_TOKEN]) for w in sentence
                ]
                padded = data_utils.pad_sentence(word_idxs, window_size)
                windows = create_windows(padded, window_size)
                X = torch.tensor(windows).to(device)
                if use_subwords:
                    pre_idxs, suf_idxs = data_utils.encode_prefix_suffix(
                        sentence, prefix2idx, suffix2idx, labeled=False
                    )
                    pre_padded = data_utils.pad_sentence(pre_idxs, window_size)
                    suf_padded = data_utils.pad_sentence(suf_idxs, window_size)
                    pre_windows = create_windows(pre_padded, window_size)
                    suf_windows = create_windows(suf_padded, window_size)
                    pre_tensor = torch.tensor(pre_windows).to(device)
                    suf_tensor = torch.tensor(suf_windows).to(device)
                    logits = model(X, pre_tensor, suf_tensor)
                else:
                    logits = model(X)
                preds = torch.argmax(logits, dim=1)
                for word, pred in zip(sentence, preds):
                    fout.write(f"{word} {idx2tag[pred.item()]}\n")
                fout.write("\n")


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
    prefix2idx=None,
    suffix2idx=None,
    use_subwords=False,
    char2idx=None,
    use_char_cnn=False,
    device="cpu",
):
    train_losses = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if use_subwords:
                X, P, S, Y = [b.to(device) for b in batch]
                logits = model(X, P, S)
            else:
                X, Y = [b.to(device) for b in batch]
                logits = model(X)
            loss = loss_fn(logits, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        dev_acc = evaluate_model(
            model,
            dev_data,
            word2idx,
            tag2idx,
            idx2tag,
            task_type,
            window_size,
            prefix2idx,
            suffix2idx,
            use_subwords,
            device,
        )
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Dev Accuracy: {dev_acc:.4f}"
        )
        train_losses.append(epoch_loss)
        dev_accuracies.append(dev_acc)

    return train_losses, dev_accuracies


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["pos", "ner"], required=True)
    parser.add_argument(
        "--part",
        type=str,
        required=True,
        help="e.g., 1 for part1, 3 for part3, 4 for part4",
    )
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument(
        "--use_pretrained_embeddings",
        action="store_true",
        help="Use pre-trained word embeddings instead of random initialization",
    )
    parser.add_argument("--use_subwords", action="store_true")
    parser.add_argument("--use_char_cnn", action="store_true")
    parser.add_argument("--max_word_len", type=int, default=42)
    parser.add_argument("--char_emb_dim", type=int, default=30)
    parser.add_argument("--num_filters", type=int, default=30)
    parser.add_argument("--filter_width", type=int, default=3)
    args = parser.parse_args()

    assert not (
        args.use_subwords and args.use_char_cnn
    ), "Cannot use both --use_subwords and --use_char_cnn at the same time."

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

    if args.use_subwords:
        prefix2idx, suffix2idx = data_utils.build_prefix_suffix_vocab(train_data)

    if args.use_char_cnn:
        char2idx = data_utils.build_char_vocab(train_data)

    # Prepare Dataloader
    all_windows, all_tags = [], []
    if args.use_subwords:
        all_prefixes, all_suffixes = [], []
    if args.use_char_cnn:
        all_char_windows = []
    for sentence in train_data:
        word_idxs, tag_idxs = data_utils.encode_sentence(sentence, word2idx, tag2idx)
        word_idxs_padded = data_utils.pad_sentence(word_idxs, args.window_size)
        windows = create_windows(word_idxs_padded, args.window_size)
        all_windows.extend(windows)
        all_tags.extend(tag_idxs)
        if args.use_subwords:
            pre_idxs, suf_idxs = data_utils.encode_prefix_suffix(
                sentence, prefix2idx, suffix2idx
            )
            padded_pre = data_utils.pad_sentence(pre_idxs, args.window_size)
            padded_suf = data_utils.pad_sentence(suf_idxs, args.window_size)
            all_prefixes += create_windows(padded_pre, args.window_size)
            all_suffixes += create_windows(padded_suf, args.window_size)
        if args.use_char_cnn:
            char_matrix = data_utils.encode_chars_per_sentence(
                sentence, char2idx, max_word_len=args.max_word_len
            )
            padded_chars = data_utils.pad_char_sentence(char_matrix, args.window_size)
            char_windows = create_windows(padded_chars, args.window_size)
            all_char_windows.extend(char_windows)

    if args.use_char_cnn:
        train_dataset = TensorDataset(
            torch.tensor(all_windows),
            torch.tensor(all_char_windows),
            torch.tensor(all_tags),
        )
    elif args.use_subwords:
        train_dataset = TensorDataset(
            torch.tensor(all_windows),
            torch.tensor(all_prefixes),
            torch.tensor(all_suffixes),
            torch.tensor(all_tags),
        )
    else:
        train_dataset = TensorDataset(torch.tensor(all_windows), torch.tensor(all_tags))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.use_char_cnn:
        model = WindowTaggerCNNSubword(
            vocab_size=len(word2idx),
            char_vocab_size=len(char2idx),
            embedding_dim=args.embedding_dim,
            char_emb_dim=args.char_emb_dim,
            num_filters=args,
            filter_width=args.filter_width,
            hidden_dim=args.hidden_dim,
            output_dim=len(tag2idx),
            window_size=args.window_size,
            max_word_len=args.max_word_len,
            pretrained_embedding=embedding_matrix,
        ).to(device)
    elif args.use_subwords:
        model = WindowTaggerSubword(
            vocab_size=len(word2idx),
            prefix_size=len(prefix2idx),
            suffix_size=len(suffix2idx),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=len(tag2idx),
            window_size=args.window_size,
            pretrained_embedding=embedding_matrix,
        ).to(device)
    else:
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
        prefix2idx if args.use_subwords else None,
        suffix2idx if args.use_subwords else None,
        args.use_subwords,
        device,
    )

    output_file_name = f"{args.task}_p{args.part}"

    # Add modifiers
    if args.use_pretrained_embeddings:
        output_file_name += "+pre"
    if args.use_subwords:
        output_file_name += "+sub"
    if args.use_char_cnn:
        output_file_name += "+cnn"

    # Save model and logs
    torch.save(model.state_dict(), f"saved_models/model_{output_file_name}.pt")
    np.save(f"logs/loss_{output_file_name}_train.npy", np.array(train_losses))
    np.save(f"logs/acc_{output_file_name}_dev.npy", np.array(dev_accuracies))

    # Plot
    plot_and_save(
        train_losses,
        f"Training Loss ({args.task})",
        "Loss",
        f"plots/loss_{output_file_name}.png",
    )
    plot_and_save(
        dev_accuracies,
        f"Dev Accuracy ({args.task})",
        "Accuracy",
        f"plots/acc_{output_file_name}.png",
    )

    # Predict
    output_file = f"predictions/test_{output_file_name}"
    predict_test(
        model,
        test_data,
        word2idx,
        idx2tag,
        output_file,
        args.window_size,
        prefix2idx if args.use_subwords else None,
        suffix2idx if args.use_subwords else None,
        args.use_subwords,
        device,
    )

    print(f"Finished training and prediction for {args.task.upper()}!")


if __name__ == "__main__":
    main()
