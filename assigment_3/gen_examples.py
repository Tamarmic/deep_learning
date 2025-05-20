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

def main():
    os.makedirs("data", exist_ok=True)

    pos_samples = [generate_positive_example() for _ in range(500)]
    neg_samples = [generate_negative_example() for _ in range(500)]

    write_examples("data/pos_examples", pos_samples)
    write_examples("data/neg_examples", neg_samples)

    print("Data generation complete. Files written to ./data/")

if __name__ == "__main__":
    main()
