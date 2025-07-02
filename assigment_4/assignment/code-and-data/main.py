import argparse
import os
import torch
from torch import optim
from transformer import TransformerLM
import data
import lm

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--embed_size", type=int, default=192)
    parser.add_argument("--mlp_hidden_size", type=int, default=768)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_batches_to_train", type=int, default=50000)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topK", type=int, default=5)
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or train tokenizer
    if os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = data.CharTokenizer.load(args.tokenizer_path)
        tokenized_data = []
        for fname in os.listdir(args.data_path):
            if fname.endswith(".txt"):
                with open(os.path.join(args.data_path, fname)) as f:
                    tokenized_data.append(tokenizer.tokenize(f.read()))
    else:
        print(f"Training tokenizer from {args.data_path}")
        tokenizer, tokenized_data = data.load_data(args.data_path)
        tokenizer.save(args.tokenizer_path)
        print(f"Tokenizer saved to {args.tokenizer_path}")

    # Data iterator
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, args.seq_len + 1))

    # Model
    model = TransformerLM(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        embed_size=args.embed_size,
        max_context_len=args.seq_len,
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=args.mlp_hidden_size,
        with_residuals=True,
        device=device
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Load checkpoint if exists
    num_batches = 0
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        num_batches = checkpoint["num_batches"]
        tokenizer = checkpoint["tokenizer"]
        print(f"Resumed from checkpoint at batch {num_batches}")

    # Training loop
    model.train()
    while num_batches < args.num_batches_to_train:
        for batch in data.batch_items(data_iter, args.batch_size):
            if num_batches >= args.num_batches_to_train:
                break

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()

            num_batches += 1

            if num_batches % 10 == 0:
                print(f"[{num_batches}] Loss: {loss.item():.4f}")

            if num_batches % 100 == 0:
                model.eval()
                prompt = "Hello"
                with torch.no_grad():
                    sampled_ids = model.better_sample_continuation(
                        tokenizer.tokenize(prompt),
                        max_tokens_to_generate=200,
                        temperature=args.temperature,
                        topK=args.topK
                    )
                    print(f"Sample: {tokenizer.detokenize(sampled_ids)}\n")
                model.train()

            # Save checkpoint
            if num_batches % 1000 == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_batches": num_batches,
                    "tokenizer": tokenizer,
                }, args.checkpoint_path)
                print(f"Checkpoint saved to {args.checkpoint_path}")

    # Save tokenizer at the end
    tokenizer.save(args.tokenizer_path)
    print(f"Tokenizer saved to {args.tokenizer_path}")
