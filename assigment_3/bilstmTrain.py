import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
from torch.utils.data import DataLoader
from data_utils import (
    read_dataset,
    build_vocab,
    encode_sentence,
    build_char_vocab,
    encode_sentence_with_chars,
    build_prefix_suffix_vocab,
    encode_prefix_suffix,
    load_embeddings,
    PAD_TOKEN,
)
from models import BiLSTMTagger


# Dataset class
class TaggingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        word2idx,
        tag2idx,
        char2idx=None,
        prefix2idx=None,
        suffix2idx=None,
        repr_mode="a",
        max_word_len=42,
    ):
        self.repr_mode = repr_mode
        self.char2idx = char2idx
        self.prefix2idx = prefix2idx
        self.suffix2idx = suffix2idx
        self.max_word_len = max_word_len
        self.data = []
        for sent in data:
            if len(sent) == 0:
                continue
            if repr_mode in ["b", "d"]:
                encoded = encode_sentence_with_chars(
                    sent, word2idx, char2idx, tag2idx, max_word_len
                )
                self.data.append(encoded)
            elif repr_mode == "c":
                word_indices, tag_indices = encode_sentence(sent, word2idx, tag2idx)
                prefix_indices, suffix_indices = encode_prefix_suffix(
                    sent, prefix2idx, suffix2idx
                )
                self.data.append(
                    (word_indices, prefix_indices, suffix_indices, tag_indices)
                )
            else:
                encoded = encode_sentence(sent, word2idx, tag2idx)
                self.data.append(encoded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Collate functions
def collate_fn_a(batch):
    words, tags = zip(*batch)
    words = [torch.tensor(s, dtype=torch.long) for s in words]
    tags = [torch.tensor(t, dtype=torch.long) for t in tags]
    lengths = [len(s) for s in words]
    padded_words = nn.utils.rnn.pad_sequence(words, batch_first=True)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0)
    return padded_words, padded_tags, torch.tensor(lengths)


def collate_fn_b(batch):
    words, chars, tags = zip(*batch)
    words = [torch.tensor(s, dtype=torch.long) for s in words]
    tags = [torch.tensor(t, dtype=torch.long) for t in tags]
    lengths = [len(s) for s in words]
    padded_words = nn.utils.rnn.pad_sequence(words, batch_first=True)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0)

    max_seq_len = max(lengths)
    max_word_len = max(len(c) for sent_chars in chars for c in sent_chars)

    B = len(chars)
    padded_chars = torch.zeros(B, max_seq_len, max_word_len, dtype=torch.long)
    char_lengths = torch.zeros(B, max_seq_len, dtype=torch.long)

    for i, sent_chars in enumerate(chars):
        for j, word_chars in enumerate(sent_chars):
            char_lengths[i, j] = len(word_chars)
            padded_chars[i, j, : len(word_chars)] = torch.tensor(
                word_chars, dtype=torch.long
            )

    return padded_words, padded_tags, torch.tensor(lengths), padded_chars, char_lengths


def collate_fn_c(batch):
    words, prefixes, suffixes, tags = zip(*batch)
    words = [torch.tensor(s, dtype=torch.long) for s in words]
    prefixes = [torch.tensor(p, dtype=torch.long) for p in prefixes]
    suffixes = [torch.tensor(suf, dtype=torch.long) for suf in suffixes]
    tags = [torch.tensor(t, dtype=torch.long) for t in tags]
    lengths = [len(s) for s in words]

    padded_words = nn.utils.rnn.pad_sequence(words, batch_first=True)
    padded_prefixes = nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
    padded_suffixes = nn.utils.rnn.pad_sequence(suffixes, batch_first=True)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0)

    return (
        padded_words,
        padded_prefixes,
        padded_suffixes,
        padded_tags,
        torch.tensor(lengths),
    )


collate_fn_d = collate_fn_b


# Accuracy helper
def compute_accuracy(logits, y, pad_idx=0, ignore_tag_idx=None):
    """
    Compute accuracy, optionally ignoring cases where both true and pred equal ignore_tag_idx.

    Args:
        logits: (B, T, C) model outputs (unnormalized)
        y: (B, T) true tag indices
        pad_idx: int, index of padding tag to ignore
        ignore_tag_idx: int or None, tag index to ignore when both pred and true equal it

    Returns:
        accuracy float
    """
    preds = torch.argmax(logits, dim=-1)
    mask = y != pad_idx
    if ignore_tag_idx is not None:
        ignore_mask = (y == ignore_tag_idx) & (preds == ignore_tag_idx)
        valid_mask = mask & (~ignore_mask)
    else:
        valid_mask = mask

    correct = (preds == y) & valid_mask
    total = valid_mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


# Train epoch function
def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
    pad_idx,
    repr_mode,
    dev_loader,
    eval_every=500,
    ignore_tag_idx=None,
):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    sentences_seen = 0
    dev_accuracies = []
    for batch in dataloader:
        if repr_mode == "a":
            x, y, lengths = batch
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)
        elif repr_mode == "b":
            x, y, lengths, chars, char_lengths = batch
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            chars, char_lengths = chars.to(device), char_lengths.to(device)
            logits = model(x, lengths, chars, char_lengths)
        elif repr_mode == "c":
            x, prefixes, suffixes, y, lengths = batch
            x, prefixes, suffixes = (
                x.to(device),
                prefixes.to(device),
                suffixes.to(device),
            )
            y, lengths = y.to(device), lengths.to(device)
            logits = model(x, prefixes, suffixes, lengths)
        elif repr_mode == "d":
            x, y, lengths, chars, char_lengths = batch
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            chars, char_lengths = chars.to(device), char_lengths.to(device)
            logits = model(x, lengths, chars, char_lengths)
        else:
            raise ValueError(f"Unknown repr_mode: {repr_mode}")

        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        loss = loss_fn(logits_flat, y_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_valid = (y != pad_idx).sum().item()
        batch_correct = compute_accuracy(logits, y, pad_idx, ignore_tag_idx) * num_valid

        total_correct += batch_correct
        total_tokens += num_valid
        total_loss += loss.item()
        sentences_seen += x.size(0)

        if sentences_seen % eval_every < x.size(0):
            dev_acc = evaluate(
                model, dev_loader, device, pad_idx, repr_mode, ignore_tag_idx
            )
            dev_accuracies.append((sentences_seen, dev_acc))
            print(f"Sentences seen: {sentences_seen}, Dev Acc: {dev_acc:.4f}")

    return total_loss / len(dataloader), total_correct / total_tokens, dev_accuracies


# Evaluation function
def evaluate(
    model,
    dataloader,
    device,
    pad_idx,
    repr_mode,
    ignore_tag_idx=None,
):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            if repr_mode == "a":
                x, y, lengths = batch
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                logits = model(x, lengths)
            elif repr_mode == "b":
                x, y, lengths, chars, char_lengths = batch
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                chars, char_lengths = chars.to(device), char_lengths.to(device)
                logits = model(x, lengths, chars, char_lengths)
            elif repr_mode == "c":
                x, prefixes, suffixes, y, lengths = batch
                x, prefixes, suffixes = (
                    x.to(device),
                    prefixes.to(device),
                    suffixes.to(device),
                )
                y, lengths = y.to(device), lengths.to(device)
                logits = model(x, prefixes, suffixes, lengths)
            elif repr_mode == "d":
                x, y, lengths, chars, char_lengths = batch
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                chars, char_lengths = chars.to(device), char_lengths.to(device)
                logits = model(x, lengths, chars, char_lengths)
            else:
                raise ValueError(f"Unknown repr_mode: {repr_mode}")

            num_valid = (y != pad_idx).sum().item()
            batch_correct = (
                compute_accuracy(logits, y, pad_idx, ignore_tag_idx) * num_valid
            )
            total_correct += batch_correct
            total_tokens += num_valid
    model.train()
    return total_correct / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repr", choices=["a", "b", "c", "d"], help="Input representation mode"
    )
    parser.add_argument("train_file", type=str, help="Training data file")
    parser.add_argument("model_file", type=str, help="Output model file")
    parser.add_argument(
        "--dev_file", type=str, default=None, help="Development data file"
    )
    parser.add_argument("--task", type=str, help="Task type (e.g., pos, ner)")
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--char_emb_dim", type=int, default=30)
    parser.add_argument("--char_hidden_dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = read_dataset(args.train_file, labeled=True)
    dev_data = read_dataset(args.dev_file, labeled=True) if args.dev_file else None

    word2idx, tag2idx, idx2tag = build_vocab(train_data)

    try:
        pretrain_word2vec = load_embeddings(
            "embeddings/vocab.txt", "embeddings/wordVectors.txt"
        )
        emb_dim = next(iter(pretrain_word2vec.values())).shape[0]
        assert (
            emb_dim == args.embedding_dim
        ), f"Embedding dimension mismatch: expected {args.embedding_dim}, got {emb_dim}"
        scale = np.sqrt(1.0 / emb_dim)
        embedding_matrix = np.random.uniform(-scale, scale, (len(word2idx), emb_dim))

        for word, idx in word2idx.items():
            if word in pretrain_word2vec:
                embedding_matrix[idx] = pretrain_word2vec[word]

        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading pretrained embeddings: {e}")
        print("Using random initialization")
        embedding_matrix = None

    char2idx = build_char_vocab(train_data) if args.repr in ["b", "d"] else None
    prefix2idx, suffix2idx = (
        build_prefix_suffix_vocab(train_data) if args.repr == "c" else (None, None)
    )

    train_set = TaggingDataset(
        train_data, word2idx, tag2idx, char2idx, prefix2idx, suffix2idx, args.repr
    )
    dev_set = (
        TaggingDataset(
            dev_data, word2idx, tag2idx, char2idx, prefix2idx, suffix2idx, args.repr
        )
        if dev_data
        else None
    )

    if args.repr == "a":
        collate_fn = collate_fn_a
    elif args.repr == "b":
        collate_fn = collate_fn_b
    elif args.repr == "c":
        collate_fn = collate_fn_c
    elif args.repr == "d":
        collate_fn = collate_fn_d

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = (
        DataLoader(
            dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        if dev_set
        else None
    )

    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        repr_mode=args.repr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        char_vocab_size=len(char2idx) if char2idx else None,
        char_emb_dim=args.char_emb_dim,
        char_hidden_dim=args.char_hidden_dim,
        prefix_size=len(prefix2idx) if prefix2idx else None,
        suffix_size=len(suffix2idx) if suffix2idx else None,
        embedding_matrix=embedding_matrix,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx[PAD_TOKEN])

    print(
        f"Training repr mode '{args.repr}' for {args.epochs} epochs on device {device} for task '{args.task}'"
    )
    ignore_tag_idx = None
    if args.task == "ner":
        ignore_tag_idx = tag2idx.get("O", None)
        if ignore_tag_idx is None:
            print("Warning: 'O' tag not found in tag2idx vocab.")
    all_dev_accuracies = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc, dev_accuracies = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            tag2idx[PAD_TOKEN],
            args.repr,
            dev_loader,
            eval_every=500,
        )
        print(
            f"Epoch {epoch+1} TRAIN loss: {train_loss:.4f} | TRAIN accuracy: {train_acc:.4f}"
        )
        all_dev_accuracies.extend(dev_accuracies)
        if dev_accuracies:
            print(f"Last dev acc: {dev_accuracies[-1][1]:.4f}")

    dev_acc_file = f"dev_acc_{args.task}_{args.repr}.pkl"
    with open(dev_acc_file, "wb") as f:
        pickle.dump(all_dev_accuracies, f)
    print(f"Dev accuracies saved to {dev_acc_file}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word2idx": word2idx,
            "tag2idx": tag2idx,
            "idx2tag": idx2tag,
            "char2idx": char2idx,
            "prefix2idx": prefix2idx,
            "suffix2idx": suffix2idx,
            "repr": args.repr,
        },
        args.model_file,
    )
    print(f"Model saved to {args.model_file}")


if __name__ == "__main__":
    main()
