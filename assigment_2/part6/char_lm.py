import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_utils import build_char_vocab, prepare_char_ngram_data, PAD_TOKEN, UNK_TOKEN
from models import CharNgramLM

def sample(model, prefix, char2idx, idx2char, k, n, device="cpu"):
    model.eval()
    result = prefix
    context = prefix[-k:].rjust(k, " ")

    for _ in range(n):
        x = torch.tensor([[char2idx.get(c, char2idx[UNK_TOKEN]) for c in context]], device=device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze()
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]
            result += next_char
            context = context[1:] + next_char

    return result


def plot_loss(train_losses, output_path):
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../lm-data/eng-data/input.txt")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--sample_prefix", type=str, default="the ")
    parser.add_argument("--sample_length", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("charlm_outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and clean text
    with open(args.file, "r", encoding="utf8") as f:
        text = f.read().lower()

    char2idx, idx2char = build_char_vocab(text)
    X, Y = prepare_char_ngram_data(text, char2idx, args.k)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CharNgramLM(
        vocab_size=len(char2idx),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        context_size=args.k,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.sample_every == 0:
            generated = sample(model, args.sample_prefix, char2idx, idx2char, args.k, args.sample_length, device)
            print(f"\nSample (epoch {epoch+1}):\n{generated}\n")

    torch.save(model.state_dict(), "charlm_outputs/char_lm.pt")
    np.save("charlm_outputs/loss_char_lm.npy", np.array(train_losses))
    plot_loss(train_losses, "charlm_outputs/loss_char_lm.png")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
