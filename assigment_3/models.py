import torch
import torch.nn as nn


class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, emb_dim, embedding_matrix=None, padding_idx=0):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)


class EmbeddingsWithPrefixSuffix(nn.Module):

    def __init__(
        self,
        vocab_size,
        prefix_size,
        suffix_size,
        emb_dim,
        embedding_matrix=None,
        padding_idx=0,
    ):
        super().__init__()
        if embedding_matrix is not None:
            self.word_emb = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False, padding_idx=padding_idx
            )
        else:
            self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.prefix_emb = nn.Embedding(prefix_size, emb_dim, padding_idx=padding_idx)
        self.suffix_emb = nn.Embedding(suffix_size, emb_dim, padding_idx=padding_idx)

    def forward(self, words, prefixes, suffixes):
        w = self.word_emb(words)
        p = self.prefix_emb(prefixes)
        s = self.suffix_emb(suffixes)
        return w + p + s


import torch
import torch.nn as nn


class CharLSTMEmbedding(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, char_hidden_dim, padding_idx=0):
        super().__init__()
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_emb_dim, padding_idx=padding_idx
        )
        self.char_lstm_cell = nn.LSTMCell(char_emb_dim, char_hidden_dim)
        self.char_hidden_dim = char_hidden_dim

    def forward(self, char_seq_padded, char_lengths, seq_lengths=None):
        B, T, max_word_len = char_seq_padded.size()
        device = char_seq_padded.device

        char_seq_flat = char_seq_padded.view(B * T, max_word_len)
        char_lengths_flat = char_lengths.view(B * T)

        h_t = torch.zeros(B * T, self.char_hidden_dim, device=device)
        c_t = torch.zeros(B * T, self.char_hidden_dim, device=device)

        for t in range(max_word_len):
            char_emb_t = self.char_embedding(char_seq_flat[:, t])
            mask = (char_lengths_flat > t).float().unsqueeze(1)
            h_t_new, c_t_new = self.char_lstm_cell(char_emb_t, (h_t, c_t))
            h_t = h_t_new * mask + h_t * (1 - mask)
            c_t = c_t_new * mask + c_t * (1 - mask)

        return h_t.view(B, T, self.char_hidden_dim)


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
    def __init__(
        self,
        vocab_size,
        tagset_size,
        repr_mode,
        embedding_dim=50,
        hidden_dim=100,
        char_vocab_size=None,
        char_emb_dim=30,
        char_hidden_dim=50,
        prefix_size=None,
        suffix_size=None,
        embedding_matrix=None,
    ):
        super().__init__()
        self.repr_mode = repr_mode
        self.hidden_dim = hidden_dim
        if repr_mode == "a":
            self.word_emb = WordEmbedding(vocab_size, embedding_dim, embedding_matrix)
            input_dim = embedding_dim
        elif repr_mode == "b":
            assert char_vocab_size is not None, "char_vocab_size required for repr b"
            self.char_emb = CharLSTMEmbedding(
                char_vocab_size, char_emb_dim, char_hidden_dim
            )
            input_dim = char_hidden_dim
        elif repr_mode == "c":
            assert (
                prefix_size is not None and suffix_size is not None
            ), "prefix and suffix sizes required for repr c"
            self.embeddings = EmbeddingsWithPrefixSuffix(
                vocab_size, prefix_size, suffix_size, embedding_dim, embedding_matrix
            )
            input_dim = embedding_dim
        elif repr_mode == "d":
            assert char_vocab_size is not None, "char_vocab_size required for repr d"
            self.word_emb = WordEmbedding(vocab_size, embedding_dim, embedding_matrix)
            self.char_emb = CharLSTMEmbedding(
                char_vocab_size, char_emb_dim, char_hidden_dim
            )
            self.linear = nn.Linear(embedding_dim + char_hidden_dim, embedding_dim)
            input_dim = embedding_dim
        else:
            raise ValueError(f"Unknown repr_mode: {repr_mode}")
        self.bilstm1 = BiLSTMLayer(input_dim, hidden_dim)
        self.bilstm2 = BiLSTMLayer(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, *args):
        if self.repr_mode == "a":
            words, lengths = args
            emb = self.word_emb(words)
        elif self.repr_mode == "b":
            words, lengths, chars, char_lengths = args
            emb = self.char_emb(chars, char_lengths, lengths)
        elif self.repr_mode == "c":
            words, prefixes, suffixes, lengths = args
            emb = self.embeddings(words, prefixes, suffixes)
        elif self.repr_mode == "d":
            words, lengths, chars, char_lengths = args
            char_repr = self.char_emb(chars, char_lengths, lengths)
            word_emb = self.word_emb(words)
            concat_emb = torch.cat([word_emb, char_repr], dim=2)
            emb = self.linear(concat_emb)
        else:
            raise ValueError(f"Unknown repr_mode: {self.repr_mode}")
        out1 = self.bilstm1(emb, lengths)
        out2 = self.bilstm2(out1, lengths)
        logits = self.classifier(out2)
        return logits
