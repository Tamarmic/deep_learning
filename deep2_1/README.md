# Window-Based Tagger and Embedding Similarity Search

## Requirements

-   PyTorch
-   Numpy
-   Matplotlib

Install missing packages:
pip install torch numpy matplotlib

## Part 1 - Window-Based Tagger

Train and predict for POS tagging:
python tagger1.py --task pos

Train and predict for NER tagging:
python tagger1.py --task ner

Arguments (default values shown):

--task Task type: 'pos' or 'ner' (default: pos)
--embedding_dim Word embedding dimension (default: 50)
--hidden_dim Hidden layer dimension (default: 100)
--batch_size Batch size for training (default: 32)
--num_epochs Number of training epochs (default: 10)
--learning_rate Learning rate (default: 0.001)
--window_size Context window size on each side (default: 2)

Outputs after training:

-   `saved_models/`: Model files
-   `logs/`: Loss and accuracy logs
-   `plots/`: Loss and accuracy graphs
-   `predictions/`: Test set predictions

## Part 2 - Most Similar Words Using Pretrained Embeddings

Run the similarity search:
python top_k.py

Arguments:

--vocab_file Path to vocab file (default: embeddings/vocab.txt)
--vectors_file Path to word vectors file (default: embeddings/wordVectors.txt)
--k Number of most similar words to retrieve (default: 5)

Outputs:

-   Console printout of top-5 most similar words (use for part2.pdf)

---
