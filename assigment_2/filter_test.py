import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from models import WindowTaggerCNNSubword
import data_utils


# --------------------------
# Load trained CNN model and vocab
# --------------------------
def load_trained_cnn_model(model_path, task, device="cpu"):
    train_data = data_utils.read_dataset(f"{task}/train")
    word2idx, tag2idx, idx2tag = data_utils.build_vocab(train_data)
    char2idx = data_utils.build_char_vocab(train_data)

    model = WindowTaggerCNNSubword(
        vocab_size=len(word2idx),
        char_vocab_size=len(char2idx),
        embedding_dim=50,
        char_emb_dim=30,
        num_filters=15,
        filter_width=3,
        hidden_dim=100,
        output_dim=len(tag2idx),
        window_size=2,
        max_word_len=42,
        pretrained_embedding=None
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, char2idx, train_data, tag2idx, idx2tag


# --------------------------
# Visualize activations of CNN filters on a word
# --------------------------
def visualize_filter_responses(model, word, char2idx, device="cpu"):
    model.eval()
    chars = list(word.lower())
    char_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in chars]
    char_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.char_embedding(char_tensor)  # (1, seq_len, emb_dim)
        emb = emb.transpose(1, 2)                # (1, emb_dim, seq_len)
        conv_out = model.char_cnn(emb)           # (1, num_filters, L_out)
        act = F.relu(conv_out).squeeze(0).cpu().numpy()  # (num_filters, L_out)

    plt.figure(figsize=(10, 6))
    plt.imshow(act, aspect="auto", cmap="gray_r")
    plt.title(f"Activations for '{word}'")
    plt.ylabel("Filter")
    plt.xlabel("Char position")
    plt.colorbar(label="Activation")
    plt.tight_layout()
    plt.show()


# --------------------------
# Get sample words for a given label (e.g. "I-LOC")
# --------------------------
def get_words_by_label(train_data, label, max_per_label=5):
    words = []
    seen = set()
    for sentence in train_data:
        for word, tag in sentence:
            if tag == label and word not in seen:
                words.append(word)
                seen.add(word)
                if len(words) >= max_per_label:
                    return words
    return words


# --------------------------
# Run full analysis
# --------------------------
def analyze(task, model_path, labels_to_check, device="cpu"):
    model, char2idx, train_data, tag2idx, idx2tag = load_trained_cnn_model(model_path, task, device)

    for label in labels_to_check:
        print(f"\n=== Activations for label: {label} ===")
        words = get_words_by_label(train_data, label)
        for word in words:
            print(f"Word: {word}")
            visualize_filter_responses(model, word, char2idx, device)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    task = "ner"  # or "pos"
    model_path = "saved_models/model_ner_p5+pre+cnn.pt"  # Change to POS model if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try these for NER; for POS use tags like "NN", "VB", etc.
    labels_to_check = ["LOC", "PER","ORG" ]

    analyze(task, model_path, labels_to_check, device)
