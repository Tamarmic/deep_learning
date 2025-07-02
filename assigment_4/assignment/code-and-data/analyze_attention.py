import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from transformer import TransformerLM
from data import CharTokenizer
import os

# Global variable to store attention weights
attention_maps = []

def get_hook(layer_idx):
    def hook(module, input, output):
        # Save [batch, heads, query_len, key_len] → remove batch dim
        attention_maps.append(module.last_attention_weights[0].detach().cpu())
    return hook

def visualize_attention(attn, tokens, layer, head):
    matrix = attn[head]  # shape: [seq_len, seq_len]
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(f'Layer {layer}, Head {head}')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.show()

def run_analysis(prompt, model_path, tokenizer_path, device):
    global attention_maps
    attention_maps = []

    # Load tokenizer
    tokenizer = CharTokenizer.load(tokenizer_path)
    tokens = tokenizer.tokenize(prompt)
    tokens_str = list(prompt)

    # Load model
    model = TransformerLM(
        n_layers=6,
        n_heads=6,
        embed_size=192,
        max_context_len=128,
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=768,
        with_residuals=True,
        device=device,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Register hooks to capture attention from each layer
    for i, block in enumerate(model.layers):
        block.causal_attention.register_forward_hook(get_hook(i))

    # Feed prompt
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        _ = model(input_ids)

    # Visualize attention from layer 0 head 0 (as example)
    for layer_idx, attn in enumerate(attention_maps):
        for head_idx in range(attn.shape[0]):
            visualize_attention(attn, tokens_str, layer=layer_idx, head=head_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="the moon is blue.")
    parser.add_argument("--model_path", type=str, default="checkpoint.pt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_analysis(args.prompt, args.model_path, args.tokenizer_path, device)
