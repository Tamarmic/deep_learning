import torch
import torch.nn as nn


class WindowTagger(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        window_size=2,
        pretrained_embedding=None,
    ):
        super(WindowTagger, self).__init__()
        self.window_size = window_size  # 2
        self.total_window_size = 2 * window_size + 1  # 5

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.fc1 = nn.Linear(
            embedding_dim * self.total_window_size, hidden_dim
        )  # [250, ?]
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, total_window_size)
        embedded = self.embedding(x)  # (batch_size, window, embed_dim)
        embedded = embedded.view(embedded.size(0), -1)  # flatten
        hidden = torch.tanh(self.fc1(embedded))  # (batch_size, hidden_dim)
        output = self.fc2(hidden)  # (batch_size, output_dim)
        return output


class WindowTaggerSubword(WindowTagger):
    def __init__(
        self,
        vocab_size,
        prefix_size,
        suffix_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        window_size,
        pretrained_embedding=None,
    ):
        super().__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            window_size,
            pretrained_embedding,
        )
        self.prefix_embedding = nn.Embedding(prefix_size, embedding_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(suffix_size, embedding_dim, padding_idx=0)

    def forward(self, x, prefix_idx, suffix_idx):
        word_emb = self.embedding(x)
        prefix_emb = self.prefix_embedding(prefix_idx)
        suffix_emb = self.suffix_embedding(suffix_idx)
        total_emb = word_emb + prefix_emb + suffix_emb
        total_emb = total_emb.view(total_emb.size(0), -1)
        hidden = torch.tanh(self.fc1(total_emb))
        return self.fc2(hidden)
