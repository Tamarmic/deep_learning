import random
import os

# Set random seed for reproducibility
random.seed(42)

def generate_segment(chars, min_len=1, max_len=10):
    """Generate a random sequence of one or more of the same character(s)."""
    return "".join(random.choices(chars, k=random.randint(min_len, max_len)))

def generate_positive_example():
    """
    Generate a valid string matching:
    [1-9]+ a+ [1-9]+ b+ [1-9]+ c+ [1-9]+ d+ [1-9]+
    """
    parts = [
        generate_segment("123456789"),
        generate_segment("a"),
        generate_segment("123456789"),
        generate_segment("b"),
        generate_segment("123456789"),
        generate_segment("c"),
        generate_segment("123456789"),
        generate_segment("d"),
        generate_segment("123456789"),
    ]
    return "".join(parts)

def generate_negative_example():
    """
    Generate an invalid string: same structure, but with c+ appearing before b+
    """
    parts = [
        generate_segment("123456789"),
        generate_segment("a"),
        generate_segment("123456789"),
        generate_segment("c"),  # c appears before b
        generate_segment("123456789"),
        generate_segment("b"),
        generate_segment("123456789"),
        generate_segment("d"),
        generate_segment("123456789"),
    ]
    return "".join(parts)

def write_examples(filename, examples):
    with open(filename, "w") as f:
        for example in examples:
            f.write(example + "\n")


###########################################################################################

def write_labeled_split(pos_samples, neg_samples, train_file, test_file, train_ratio=0.8):
    # Create labeled dataset
    labeled = [(s, 1) for s in pos_samples] + [(s, 0) for s in neg_samples]
    random.shuffle(labeled)

    split_idx = int(len(labeled) * train_ratio)
    train_data = labeled[:split_idx]
    test_data = labeled[split_idx:]

    with open(train_file, "w") as f:
        for seq, label in train_data:
            f.write(f"{seq}\t{label}\n")

    with open(test_file, "w") as f:
        for seq, label in test_data:
            f.write(f"{seq}\t{label}\n")

# def main():
#     os.makedirs("data", exist_ok=True)
#
#     pos_samples = [generate_positive_example() for _ in range(500)]
#     neg_samples = [generate_negative_example() for _ in range(500)]
#
#     split_and_save(pos_samples, "pos", "data")
#     split_and_save(neg_samples, "neg", "data")
#
#     print("Train/test split complete. Files written to ./data/")

# if __name__ == "__main__":
#     main()

###########################################################################################

def main():
    os.makedirs("data", exist_ok=True)

    pos_samples = [generate_positive_example() for _ in range(5000)]
    neg_samples = [generate_negative_example() for _ in range(5000)]

    # write_examples("data/pos_examples", pos_samples)
    # write_examples("data/neg_examples", neg_samples)

    write_labeled_split(pos_samples, neg_samples, "data/train.txt", "data/test.txt")
    print("Generated train.txt and test.txt with labeled sequences.")



if __name__ == "__main__":
    main()
