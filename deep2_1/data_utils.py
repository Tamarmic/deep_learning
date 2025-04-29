import numpy as np

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
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip().lower())

    vectors = np.loadtxt(vectors_file)
    assert len(vocab) == vectors.shape[0], "Vocab and vectors size mismatch."

    word_to_vec = {word: vec for word, vec in zip(vocab, vectors)}
    return word_to_vec
