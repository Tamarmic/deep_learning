import matplotlib.pyplot as plt
import torch
from torch import Tensor
from pathlib import Path

from transformer import TransformerLM
from attention import kqv, attention_scores
from data import CharTokenizer


def heat_map_kqv(x: Tensor, kqv_matrix: Tensor) -> Tensor:
    k, q, v = kqv(x, kqv_matrix)
    return attention_scores(k, q).detach().cpu()


def create_headwise_plots(model: TransformerLM, x: Tensor, token_labels: list[str], save_dir: str):
    Path(save_dir).mkdir(exist_ok=True)
    layer = model.layers[0]  # Only layer 1

    for j, head_matrix in enumerate(layer.causal_attention.kqv_matrices):
        heat_map = heat_map_kqv(x, head_matrix)[0]  # [B, T, T] → take batch=0

        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(heat_map.numpy(), cmap="Greys")
        fig.colorbar(cax, ax=ax)

        ax.set_title(f"Layer 1, Head {j + 1}")
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90)
        ax.set_yticklabels(token_labels)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/layer1_head{j + 1}.png")
        plt.close()


# Run setup
model_path = "checkpoint.pt"
tokenizer_path = "tokenizer.json"
prompt = "If a then b. If b then c. If c then d. If d then e. This creates a chain of logic."
max_len = 128
save_dir = "layer1_attention_pages"

# Load tokenizer and model
tokenizer = CharTokenizer.load(tokenizer_path)
token_ids = tokenizer.tokenize(prompt)
token_labels = list(prompt)
tokens_tensor = torch.tensor([token_ids], dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerLM(
    n_layers=6,
    n_heads=6,
    embed_size=192,
    max_context_len=max_len,
    vocab_size=tokenizer.vocab_size(),
    mlp_hidden_size=4 * 192,
    with_residuals=True,
    device=device,
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
model.eval()

# Generate plots
with torch.no_grad():
    embeddings = model.embed(tokens_tensor.to(device))
    create_headwise_plots(model, embeddings, token_labels, save_dir)
