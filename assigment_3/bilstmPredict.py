import torch
import argparse
from data_utils import (
    encode_sentence,
    encode_sentence_with_chars,
    encode_prefix_suffix,
)
from models import BiLSTMTagger


class BiLSTMTaggerPredictor:
    def __init__(self, model_file, device=None):
        checkpoint = torch.load(model_file, map_location="cpu")
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.word2idx = checkpoint["word2idx"]
        self.tag2idx = checkpoint["tag2idx"]
        self.idx2tag = checkpoint["idx2tag"]
        self.char2idx = checkpoint.get("char2idx", None)
        self.prefix2idx = checkpoint.get("prefix2idx", None)
        self.suffix2idx = checkpoint.get("suffix2idx", None)
        self.repr_mode = checkpoint["repr"]

        char_vocab_size = len(self.char2idx) if self.char2idx else None
        prefix_size = len(self.prefix2idx) if self.prefix2idx else None
        suffix_size = len(self.suffix2idx) if self.suffix2idx else None

        self.model = BiLSTMTagger(
            vocab_size=len(self.word2idx),
            tagset_size=len(self.tag2idx),
            repr_mode=self.repr_mode,
            char_vocab_size=char_vocab_size,
            prefix_size=prefix_size,
            suffix_size=suffix_size,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_sentence(self, sentence):
        # sentence: list of (word,) or (word, tag) tuples
        if self.repr_mode == "a":
            words_idx, _ = encode_sentence(sentence, self.word2idx, self.tag2idx)
            x = torch.tensor([words_idx], dtype=torch.long).to(self.device)
            lengths = torch.tensor([len(words_idx)], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.model(x, lengths)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        elif self.repr_mode == "b":
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
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
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
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        elif self.repr_mode == "d":
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
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        else:
            raise ValueError(f"Unknown repr_mode: {self.repr_mode}")
        pred_tags = [self.idx2tag[idx] for idx in preds[: len(sentence)]]
        return pred_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str, help="Trained model file")
    parser.add_argument(
        "input_file", type=str, help="Input file for prediction (one sentence per line)"
    )
    parser.add_argument("output_file", type=str, help="Output file for predicted tags")
    args = parser.parse_args()

    predictor = BiLSTMTaggerPredictor(args.model_file)

    with open(args.input_file, "r", encoding="utf-8") as fin, open(
        args.output_file, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            words = line.strip().split()
            sentence = [(w,) for w in words]
            pred_tags = predictor.predict_sentence(sentence)
            fout.write(" ".join(pred_tags) + "\n")
