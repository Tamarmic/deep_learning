import numpy as np
import torch

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def read_dataset(file_path, labeled=True):
    """
    Reads a dataset file:
    - If labeled=True, expects 'word tag' per line.
    - If labeled=False, expects only 'word' per line (no tags).
    Sentences are separated by blank lines.
    """
    data = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    data.append(current_sentence)
                    current_sentence = []
            else:
                if labeled:
                    word, tag = line.split()
                    word = word.lower()
                    current_sentence.append((word, tag))
                else:
                    word = line.lower()
                    current_sentence.append(word)
        if current_sentence:
            data.append(current_sentence)

    return data


def build_vocab(data):
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    tag2idx = {}
    idx2tag = {}

    for sentence in data:
        for word, tag in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            if tag not in tag2idx:
                idx2tag[len(tag2idx)] = tag
                tag2idx[tag] = len(tag2idx)

    return word2idx, tag2idx, idx2tag


def encode_sentence(sentence, word2idx, tag2idx):
    """
    Converts a sentence of (word, tag) into two lists of indices.
    Unknown words are replaced with UNK index.
    """
    word_indices = [word2idx.get(word, word2idx[UNK_TOKEN]) for word, _ in sentence]
    tag_indices = [tag2idx[tag] for _, tag in sentence]
    return word_indices, tag_indices


def pad_sentence(word_indices, window_size=2):
    """
    Pads the sentence with PAD_TOKEN indices for window-based models.
    """
    pad = [0] * window_size  # 0 is the index for <PAD>
    return pad + word_indices + pad


def load_embeddings(vocab_file, vectors_file):
    """
    Loads pre-trained embeddings.
    Returns: dict {word: vector}
    """
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            vocab.append(line.strip().lower())

    vectors = np.loadtxt(vectors_file)
    assert len(vocab) == vectors.shape[0], "Vocab and vectors size mismatch."

    word_to_vec = {word: vec for word, vec in zip(vocab, vectors)}
    return word_to_vec


def build_prefix_suffix_vocab(data):
    prefix2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    suffix2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for sentence in data:
        for word, _ in sentence:
            prefix = word[:3]
            suffix = word[-3:]
            if prefix not in prefix2idx:
                prefix2idx[prefix] = len(prefix2idx)
            if suffix not in suffix2idx:
                suffix2idx[suffix] = len(suffix2idx)
    return prefix2idx, suffix2idx


def encode_prefix_suffix(sentence, prefix2idx, suffix2idx, labeled=True):
    """
    Converts a sentence of (word, tag) into two lists of prefix and suffix indices.
    Unknown prefixes/suffixes are replaced with UNK index.
    """
    if labeled:
        sentence = [word for word, _ in sentence]
    prefix_indices = [
        prefix2idx.get(word[:3], prefix2idx[UNK_TOKEN]) for word in sentence
    ]
    suffix_indices = [
        suffix2idx.get(word[-3:], suffix2idx[UNK_TOKEN]) for word in sentence
    ]
    return prefix_indices, suffix_indices

# part6
def prepare_char_ngram_data(text, char2idx, k):
    X = []
    Y = []

    for i in range(k, len(text)):
        context = text[i - k : i]
        target = text[i]

        x_indices = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in context]
        y_index = char2idx.get(target, char2idx[UNK_TOKEN])

        X.append(x_indices)
        Y.append(y_index)

    return torch.tensor(X), torch.tensor(Y)

def build_char_vocab(text):
    chars = sorted(set(text))
    char2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for c in chars:
        if c not in char2idx:
            char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char