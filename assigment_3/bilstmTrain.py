import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from data_utils import read_dataset, build_vocab, encode_sentence

# --------------------------
# Dataset and Collate
# --------------------------
class TaggingDataset(torch.utils.data.Dataset):
    def __init__(self, data, word2idx, tag2idx):
        self.data = [encode_sentence(sent, word2idx, tag2idx) for sent in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    labels = [torch.tensor(lbl, dtype=torch.long) for lbl in labels]  # ✅ this line
    lengths = [len(seq) for seq in sequences]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)  # 0 = <PAD>
    return padded_sequences, padded_labels, torch.tensor(lengths)

# --------------------------
# 2-layer BiLSTM with LSTMCell
# --------------------------
class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fwd_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.bwd_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, lengths):
        B, T, D = x.size()
        device = x.device

        h_fwd = torch.zeros(B, self.hidden_dim, device=device)
        c_fwd = torch.zeros(B, self.hidden_dim, device=device)
        fwd_out = []

        for t in range(T):
            emb_t = x[:, t]
            mask = (lengths > t).float().unsqueeze(1)
            h_new, c_new = self.fwd_cell(emb_t, (h_fwd, c_fwd))
            h_fwd = h_new * mask + h_fwd * (1 - mask)
            c_fwd = c_new * mask + c_fwd * (1 - mask)
            fwd_out.append(h_fwd.unsqueeze(1))

        fwd_out = torch.cat(fwd_out, dim=1)

        h_bwd = torch.zeros(B, self.hidden_dim, device=device)
        c_bwd = torch.zeros(B, self.hidden_dim, device=device)
        bwd_out = [None] * T

        for t in reversed(range(T)):
            emb_t = x[:, t]
            mask = (lengths > t).float().unsqueeze(1)
            h_new, c_new = self.bwd_cell(emb_t, (h_bwd, c_bwd))
            h_bwd = h_new * mask + h_bwd * (1 - mask)
            c_bwd = c_new * mask + c_bwd * (1 - mask)
            bwd_out[t] = h_bwd.unsqueeze(1)

        bwd_out = torch.cat(bwd_out, dim=1)
        return torch.cat([fwd_out, bwd_out], dim=-1)

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm1 = BiLSTMLayer(emb_dim, hidden_dim)
        self.bilstm2 = BiLSTMLayer(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        out1 = self.bilstm1(emb, lengths)
        out2 = self.bilstm2(out1, lengths)
        return self.classifier(out2)

# --------------------------
# Accuracy Helper
# --------------------------
def compute_accuracy(logits, y, pad_idx=0):
    preds = torch.argmax(logits, dim=-1)
    mask = (y != pad_idx)
    correct = (preds == y) & mask
    return correct.sum().item() / mask.sum().item()

# --------------------------
# Training and Evaluation
# --------------------------
def train_epoch(model, dataloader, optimizer, loss_fn, device, pad_idx, dev_loader=None, eval_every=500):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    num_seen = 0

    for x, y, lengths in dataloader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        logits = model(x, lengths)
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        loss = loss_fn(logits_flat, y_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_valid = (y != pad_idx).sum().item()
        batch_correct = compute_accuracy(logits, y, pad_idx) * num_valid

        total_correct += batch_correct
        total_tokens += num_valid
        total_loss += loss.item()
        num_seen += x.size(0)

        if dev_loader and num_seen % eval_every < x.size(0):
            dev_acc = evaluate(model, dev_loader, device, pad_idx)
            train_acc = total_correct / total_tokens
            print(f"[{num_seen} samples] Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f}")

    return total_loss / len(dataloader), total_correct / total_tokens

def evaluate(model, dataloader, device, pad_idx):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y, lengths in dataloader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)
            num_valid = (y != pad_idx).sum().item()
            batch_correct = compute_accuracy(logits, y, pad_idx) * num_valid
            total_correct += batch_correct
            total_tokens += num_valid
    return total_correct / total_tokens

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr", choices=["a"], default="a")
    parser.add_argument("--train_file", type=str, default="ner/train")
    parser.add_argument("--dev_file", type=str, default="ner/dev")
    parser.add_argument("--model_file", type=str, default="model.pt")
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = read_dataset(args.train_file, labeled=True)
    dev_data = read_dataset(args.dev_file, labeled=True)
    word2idx, tag2idx, idx2tag = build_vocab(train_data)
    pad_idx = tag2idx["<PAD>"]

    train_set = TaggingDataset(train_data, word2idx, tag2idx)
    dev_set = TaggingDataset(dev_data, word2idx, tag2idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMTagger(len(word2idx), args.embedding_dim, args.hidden_dim, len(tag2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device, pad_idx, dev_loader, eval_every=500)
        dev_acc = evaluate(model, dev_loader, device, pad_idx)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f} | Loss: {loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx,
        "tag2idx": tag2idx,
        "idx2tag": idx2tag
    }, args.model_file)
    print(f"Model saved to {args.model_file}")

if __name__ == "__main__":
    main()
