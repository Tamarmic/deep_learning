import torch
import argparse
from data_utils import (
    encode_sentence,
    encode_sentence_with_chars,
    encode_prefix_suffix,
    read_dataset,
    PAD_TOKEN,
)
from models import BiLSTMTagger
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


class BiLSTMTaggerPredictor:
    def __init__(self, model_file, repr_mode, device=None):
        checkpoint = torch.load(model_file, map_location="cpu")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.repr_mode = checkpoint["repr"]
        if self.repr_mode != repr_mode:
            raise ValueError(
                f"Model was trained with repr '{self.repr_mode}', got '{repr_mode}'"
            )

        self.word2idx = checkpoint["word2idx"]
        self.tag2idx = checkpoint["tag2idx"]
        self.idx2tag = checkpoint["idx2tag"]
        self.char2idx = checkpoint.get("char2idx", None)
        self.prefix2idx = checkpoint.get("prefix2idx", None)
        self.suffix2idx = checkpoint.get("suffix2idx", None)

        char_vocab_size = len(self.char2idx) if self.char2idx else None
        prefix_size = len(self.prefix2idx) if self.prefix2idx else None
        suffix_size = len(self.suffix2idx) if self.suffix2idx else None

        self.model = BiLSTMTagger(
            vocab_size=len(self.word2idx),
            tagset_size=len(self.tag2idx),
            repr_mode=self.repr_mode,
            embedding_dim=checkpoint.get("embedding_dim", 50),
            hidden_dim=checkpoint.get("hidden_dim", 100),
            char_vocab_size=char_vocab_size,
            char_emb_dim=checkpoint.get("char_emb_dim", 30),
            char_hidden_dim=checkpoint.get("char_hidden_dim", 50),
            prefix_size=prefix_size,
            suffix_size=suffix_size,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_sentence(self, sentence):
        # sentence: list of (word,)
        if self.repr_mode == "a":
            words_idx, _ = encode_sentence(sentence, self.word2idx, self.tag2idx)
            x = torch.tensor([words_idx], dtype=torch.long).to(self.device)
            lengths = torch.tensor([len(words_idx)], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.model(x, lengths)

        elif self.repr_mode == "b" or self.repr_mode == "d":
            words_idx, chars_idx, _ = encode_sentence_with_chars(
                sentence, self.word2idx, self.char2idx, self.tag2idx
            )
            x = torch.tensor([words_idx], dtype=torch.long).to(self.device)
            lengths = torch.tensor([len(words_idx)], dtype=torch.long).to(self.device)
            max_word_len = max(len(c) for c in chars_idx)
            chars_padded = torch.zeros(
                1, lengths.item(), max_word_len, dtype=torch.long
            )
            char_lengths = torch.zeros(1, lengths.item(), dtype=torch.long)
            for i, c in enumerate(chars_idx):
                chars_padded[0, i, : len(c)] = torch.tensor(c)
                char_lengths[0, i] = len(c)
            chars_padded = chars_padded.to(self.device)
            char_lengths = char_lengths.to(self.device)
            with torch.no_grad():
                logits = self.model(x, lengths, chars_padded, char_lengths)

        elif self.repr_mode == "c":
            words_idx, _ = encode_sentence(sentence, self.word2idx, self.tag2idx)
            prefix_idx, suffix_idx = encode_prefix_suffix(
                sentence, self.prefix2idx, self.suffix2idx
            )
            x = torch.tensor([words_idx], dtype=torch.long).to(self.device)
            prefixes = torch.tensor([prefix_idx], dtype=torch.long).to(self.device)
            suffixes = torch.tensor([suffix_idx], dtype=torch.long).to(self.device)
            lengths = torch.tensor([len(words_idx)], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.model(x, prefixes, suffixes, lengths)

        else:
            raise ValueError(f"Unknown repr_mode: {self.repr_mode}")

        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pred_tags = [self.idx2tag[idx] for idx in preds[: len(sentence)]]
        return pred_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repr", choices=["a", "b", "c", "d"], help="Input representation mode"
    )
    parser.add_argument("model_file", type=str, help="Trained model file")
    parser.add_argument("input_file", type=str, help="Input CoNLL test file")
    parser.add_argument("--output_file", type=str, default="predictions.txt")
    args = parser.parse_args()

    predictor = BiLSTMTaggerPredictor(args.model_file, args.repr)

    # Read unlabeled data
    sentences = read_dataset(args.input_file, labeled=False)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for sentence in sentences:
            sentence_tuples = [(w, PAD_TOKEN) for w in sentence]
            pred_tags = predictor.predict_sentence(sentence_tuples)
            for word, tag in zip(sentence, pred_tags):
                fout.write(f"{word} {tag}\n")
            fout.write("\n")
