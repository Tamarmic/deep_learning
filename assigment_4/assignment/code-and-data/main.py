from __future__ import annotations

import argparse

import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from transformer import TransformerLM
    import data
    import lm

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--embed_size", type=int, default=192)
    parser.add_argument("--mlp_hidden_size", type=int, default=768)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_batches_to_train", type=int, default=50000)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--dropout", type=float, default=0.1)  # dropout for model layers
    args = parser.parse_args()

    # seq_len = 128
    # batch_size = 64
    # data_path = "data/"
    # n_layers = 6
    # n_heads = 6
    # embed_size = 192
    # mlp_hidden_size = embed_size * 4
    #
    # learning_rate = 5e-4
    # weight_decay = 1e-5
    # gradient_clipping = 1.0
    #
    # num_batches_to_train = 50000
    seq_len = args.seq_len
    batch_size = args.batch_size
    data_path = args.data_path
    n_layers = args.n_layers
    n_heads = args.n_heads
    embed_size = args.embed_size
    mlp_hidden_size = args.embed_size * 4

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    gradient_clipping = args.gradient_clipping

    num_batches_to_train = args.num_batches_to_train

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = True,
            device=device,
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)



    model.train()
    
    num_batches = 0
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train: break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(model.sample_continuation(tokenizer.tokenize("Hello"), 500))
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")
