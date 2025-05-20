import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import argparse

# --------------------------
# Utilities
# --------------------------
def build_char_vocab(examples):
    chars = sorted(set("".join(examples)))
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for ch in chars:
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)
    return char2idx

def encode_sequence(seq, char2idx):
    return [char2idx.get(ch, char2idx["<UNK>"]) for ch in seq]

def load_labeled_data(path):
    sequences, labels = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sequences.append(parts[0])
                labels.append(int(parts[1]))
    return sequences, labels

# --------------------------
# Dataset
# --------------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, char2idx):
        self.data = [encode_sequence(seq, char2idx) for seq in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(lengths), torch.stack(labels)

# --------------------------
# LSTMCell-Based Model
# --------------------------
class LSTMAcceptor(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(emb_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths):
        emb = self.embedding(x)  # (B, T, D)
        B, T, D = emb.size()
        h_t = torch.zeros(B, self.lstm_cell.hidden_size, device=x.device)
        c_t = torch.zeros(B, self.lstm_cell.hidden_size, device=x.device)

        for t in range(T):
            emb_t = emb[:, t]  # (B, D)
            h_t, c_t = self.lstm_cell(emb_t, (h_t, c_t))

        return self.classifier(h_t).squeeze(1)

# --------------------------
# Training & Evaluation
# --------------------------
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, lengths, y in dataloader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total

# --------------------------
# Main Function
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.txt")
    parser.add_argument("--test_file", type=str, default="data/test.txt")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process data
    train_x, train_y = load_labeled_data(args.train_file)
    test_x, test_y = load_labeled_data(args.test_file)
    char2idx = build_char_vocab(train_x + test_x)

    train_data = SequenceDataset(train_x, train_y, char2idx)
    test_data = SequenceDataset(test_x, test_y, char2idx)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LSTMAcceptor(len(char2idx), args.embedding_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train loop
    print("Training started...")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f} - Test Acc: {acc:.4f}")
    end_time = time.time()

    # Final report
    print(f"\n✅ Finished in {end_time - start_time:.2f} seconds.")
    print(f"✔️  Final Test Accuracy: {evaluate(model, test_loader, device):.4f}")
    print(f"✔️  Total training steps: {len(train_loader) * args.num_epochs}")

if __name__ == "__main__":
    main()
