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
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.data = [encode_sentence(sent, word2idx, tag2idx) for sent in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.stack(labels), torch.tensor(lengths)

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

        # Forward
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

        fwd_out = torch.cat(fwd_out, dim=1)  # (B, T, H)

        # Backward
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

        bwd_out = torch.cat(bwd_out, dim=1)  # (B, T, H)

        return torch.cat([fwd_out, bwd_out], dim=-1)  # (B, T, 2H)

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.bilstm1 = BiLSTMLayer(emb_dim, hidden_dim)
        self.bilstm2 = BiLSTMLayer(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, lengths):
        emb = self.embedding(x)  # (B, T, D)
        out1 = self.bilstm1(emb, lengths)    # (B, T, 2H)
        out2 = self.bilstm2(out1, lengths)   # (B, T, 2H)
        return self.classifier(out2)         # (B, T, num_tags)


# --------------------------
# Training Function
# --------------------------
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y, lengths in dataloader:
        num_samples = 0
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)  # (B, T, C)
        logits = logits.view(-1, logits.size(-1))  # (B*T, C)
        y = y.view(-1)  # (B*T)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_samples += x.size(0)
        if num_samples >= 500:

    return total_loss / len(dataloader)

# --------------------------
# Main Script
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repr", choices=["a"], default="a")  # currently only mode (a)
    parser.add_argument("train_file", type=str, default="ner/train")
    parser.add_argument("dev_file", type=str, default="ner/dev")
    parser.add_argument("model_file", type=str)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = read_dataset(args.train_file, labeled=True)
    word2idx, tag2idx, idx2tag = build_vocab(data)
    dataset = TaggingDataset(data, word2idx, tag2idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(tag2idx),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

    # Save model and vocabs
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx,
        "tag2idx": tag2idx,
        "idx2tag": idx2tag
    }, args.model_file)
    print(f"Model saved to {args.model_file}")

if __name__ == "__main__":
    main()
