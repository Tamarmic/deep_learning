import numpy as np
import argparse
import os

import data_utils

def cosine_similarity(u, v):
    """Compute cosine similarity between two vectors."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def most_similar(word, word_to_vec, k=5):
    """Find top-k most similar words to the given word."""
    word = word.lower()  # force lowercase
    if word not in word_to_vec:
        raise ValueError(f"Word '{word}' not found in the vocabulary.")

    word_vec = word_to_vec[word]
    similarities = {}

    for other_word, other_vec in word_to_vec.items():
        if other_word == word:
            continue
        sim = cosine_similarity(word_vec, other_vec)
        similarities[other_word] = sim

    top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    return top_k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default="embeddings/vocab.txt",
                        help="Path to vocab.txt file")
    parser.add_argument("--vectors_file", type=str, default="embeddings/wordVectors.txt",
                        help="Path to wordVectors.txt file")
    parser.add_argument("--k", type=int, default=5, help="Top-k most similar words")
    args = parser.parse_args()

    if not os.path.exists(args.vocab_file) or not os.path.exists(args.vectors_file):
        raise FileNotFoundError("Vocab or vector file not found. Check the provided paths.")

    word_to_vec = data_utils.load_embeddings(args.vocab_file, args.vectors_file)

    query_words = ["dog", "england", "john", "explode", "office"]

    for query_word in query_words:
        print(f"\nTop {args.k} similar words for '{query_word}':")
        try:
            top_words = most_similar(query_word, word_to_vec, k=args.k)
            for word, sim in top_words:
                print(f"{word}: {sim:.4f}")
        except ValueError as e:
            print(e)
