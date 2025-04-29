# Assigment 2

## Requirements

-   PyTorch
-   Numpy
-   Matplotlib

Install missing packages:
pip install torch numpy matplotlib

## Part 1 - Window-Based Tagger (Random Embeddings)

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
--use_pretrained_embeddings Use external embeddings instead of random (Part 3)

Outputs after training:

-   `saved_models/`: Trained model files
-   `logs/`: Loss and accuracy logs (.npy)
-   `plots/`: Loss and accuracy graphs
-   `predictions/`: Predicted tags for test sets (e.g., test1.pos, test3.ner)

---

## Part 2 - Most Similar Words Using Pretrained Embeddings

Run similarity search:
python top_k.py

Arguments:

--vocab_file Path to vocab file (default: embeddings/vocab.txt)  
--vectors_file Path to word vectors file (default: embeddings/wordVectors.txt)  
--k Number of most similar words to retrieve (default: 5)

Outputs:

-   Console output: top-k similar words for several test cases
-   Use this output in part2.pdf

---

## Part 3 - Window-Based Tagger with Pretrained Embeddings

Train model using pretrained embeddings:
python tagger1.py --task pos --use_pretrained_embeddings

The script will automatically load embeddings from:

-   `embeddings/vocab.txt`
-   `embeddings/wordVectors.txt`

Outputs:

-   Saved model: `saved_models/model3_pos.pt`
-   Logs: `logs/loss3_pos_train.npy`, etc.
-   Predictions: `predictions/test3.pos` or `test3.ner`
-   Plots: `plots/loss3_pos.png`, `plots/acc3_pos.png`, etc.

---
