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



class WindowTaggerCNNSubword(nn.Module):
    def __init__(
        self,
        vocab_size,
        char_vocab_size,
        embedding_dim,
        char_emb_dim,
        num_filters,
        filter_width,
        hidden_dim,
        output_dim,
        window_size,
        max_word_len,
        pretrained_embedding=None,
    ):
        super(WindowTaggerCNNSubword, self).__init__()
        self.window_size = window_size
        self.total_window_size = 2 * window_size + 1
        self.max_word_len = max_word_len

        # Word embedding layer
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Character-level CNN embedding
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=num_filters,
            kernel_size=filter_width,
            padding=filter_width - 1,
        )
        # Final classification MLP
        combined_dim = embedding_dim + num_filters
        self.fc1 = nn.Linear(combined_dim * self.total_window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, word_windows, char_windows):
        # word_windows: (batch, window)
        # char_windows: (batch, window, max_word_len)

        batch_size = word_windows.size(0)
        window_size = word_windows.size(1)

        # Word embeddings
        word_emb = self.embedding(word_windows)  # (batch, window, embedding_dim)

        # Character CNN encoding
        char_windows = char_windows.view(batch_size * window_size, self.max_word_len)
        char_emb = self.char_embedding(char_windows)  # (B*W, max_len, char_emb_dim)
        char_emb = char_emb.transpose(1, 2)  # (B*W, char_emb_dim, max_len)
        conv_out = self.char_cnn(char_emb)  # (B*W, filters, L)
        pooled = torch.max(conv_out, dim=2)[0]  # (B*W, filters)
        pooled = pooled.view(batch_size, window_size, -1)  # (B, window, filters)

        # Combine word and character embeddings
        combined = torch.cat([word_emb, pooled], dim=2)  # (B, window, emb + filters)
        combined = combined.view(batch_size, -1)  # flatten: (B, total_input_dim)

        hidden = torch.tanh(self.fc1(combined))  # (B, hidden_dim)
        return self.fc2(hidden)


# part6
class CharNgramLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(context_size * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  # (batch_size, k, emb_dim)
        emb = emb.view(emb.size(0), -1)
        h = torch.tanh(self.fc1(emb))
        return self.fc2(h)