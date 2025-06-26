import random
import os

random.seed(42)

def repeat_char(char, n=None):
    if n is None:
        n = random.randint(1, 10)
    return char * n

def generate_language1_positive():
    """Pattern: [1-9]+ a+ [1-9]+ b+ [1-9]+ c+ [1-9]+ d+ [1-9]+"""
    return (
        repeat_char("123456789"),
        repeat_char("a"),
        repeat_char("123456789"),
        repeat_char("b"),
        repeat_char("123456789"),
        repeat_char("c"),
        repeat_char("123456789"),
        repeat_char("d"),
        repeat_char("123456789"),
    )

def generate_language1_negative():
    """Negative: [1-9]+ a+ [1-9]+ c+ [1-9]+ b+ [1-9]+ d+ [1-9]+"""
    return (
        repeat_char("123456789"),
        repeat_char("a"),
        repeat_char("123456789"),
        repeat_char("c"),
        repeat_char("123456789"),
        repeat_char("b"),
        repeat_char("123456789"),
        repeat_char("d"),
        repeat_char("123456789"),
    )

def generate_language2_positive():
    """Pattern: a^n b^n (e.g., aaa bbb, aa bb, etc.)"""
    n = random.randint(1, 50)
    return ("", repeat_char("a", n), repeat_char("b", n), "")

def generate_language2_negative():
    """Wrong count mismatch: a^n b^m with n != m"""
    n = random.randint(1, 50)
    m = random.choice([i for i in range(1, 50) if i != n])
    return ("", repeat_char("a", n), repeat_char("b", m), "")

def generate_language3_positive():
    """Pattern: <count>a <count>b <count>c (e.g., 3aaa2bb5ccccc)"""
    a_n = random.randint(1, 9)
    b_n = random.randint(1, 9)
    c_n = random.randint(1, 9)
    d_n = random.randint(1, 9)
    return (str(a_n), "a" * a_n, str(b_n), "b" * b_n, str(c_n), "c" * c_n, str(d_n), "d" * d_n)

def generate_language3_negative():
    """Wrong counts: 3aaa2bbb1c (count doesn't match repetition)"""
    a_n = random.randint(1, 9)
    b_n = random.randint(1, 9)
    c_n = random.randint(1, 9)
    d_n = random.randint(1, 9)
    return (str(a_n), "a" * random.randint(1, 9),
            str(b_n), "b" * random.randint(1, 9),
            str(c_n), "c" * random.randint(1, 9),
            str(b_n), "d" * random.randint(1, 9),)

import string

def generate_language4_positive(min_len=3, max_len=10):
    """
    Generate palindrome strings of lowercase letters (e.g., 'abcba', 'radar').
    Length is random between min_len and max_len.
    """
    length = random.randint(min_len, max_len)
    half = length // 2
    middle = length % 2

    # Generate random half string
    half_str = [random.choice(string.ascii_lowercase) for _ in range(half)]
    middle_str = [random.choice(string.ascii_lowercase)] if middle else []

    # Construct palindrome: half + middle + reversed half
    palindrome = half_str + middle_str + half_str[::-1]
    return ("".join(palindrome),)

def generate_language4_negative(min_len=3, max_len=10):
    """
    Generate strings that are almost palindromes but differ at one random position,
    so they are NOT palindromes.
    """
    length = random.randint(min_len, max_len)
    half = length // 2
    middle = length % 2

    half_str = [random.choice(string.ascii_lowercase) for _ in range(half)]
    middle_str = [random.choice(string.ascii_lowercase)] if middle else []

    palindrome = half_str + middle_str + half_str[::-1]

    # Change one random position in the string to break palindrome
    idx_to_change = random.randint(0, length - 1)
    original_char = palindrome[idx_to_change]
    new_char = random.choice([c for c in string.ascii_lowercase if c != original_char])
    palindrome = list(palindrome)
    palindrome[idx_to_change] = new_char
    return ("".join(palindrome),)

def generate_language5_positive():
    """
    Pattern: a^n b^k c^n d^k
    n,k random positive ints independently
    """
    n = random.randint(1, 10)
    k = random.randint(1, 10)
    return ("a" * n, "b" * k, "c" * n, "d" * k)

def generate_language5_negative():
    """
    Negative: break either counts of a/c or b/d to not match
    Example: a^(n) b^(k) c^(n') d^(k) or a^(n) b^(k) c^(n) d^(k')
    """
    n = random.randint(1, 10)
    k = random.randint(1, 10)

    # Choose which pair to break
    break_ac = random.choice([True, False])

    if break_ac:
        n2 = random.choice([x for x in range(1, 11) if x != n])
        return ("a" * n, "b" * k, "c" * n2, "d" * k)
    else:
        k2 = random.choice([x for x in range(1, 11) if x != k])
        return ("a" * n, "b" * k, "c" * n, "d" * k2)
def generate_language6_positive():
    """
    Pattern: a^(even) b^(even) c^(even)
    Each count is a positive even integer (2, 4, 6, ...)
    """
    m = random.randint(1, 5)
    n = random.randint(1, 5)
    k = random.randint(1, 5)
    return ("a" * (2 * m), "b" * (2 * n), "c" * (2 * k))

def generate_language6_negative():
    """
    Negative example: at least one character count is odd
    """
    def random_count():
        # return either odd or even count randomly
        return random.choice([2 * random.randint(1, 5), 2 * random.randint(1, 5) - 1])

    a_count = random_count()
    b_count = random_count()
    c_count = random_count()

    # Ensure at least one is odd (negative)
    while a_count % 2 == 0 and b_count % 2 == 0 and c_count % 2 == 0:
        # if all even, force one to be odd
        choice_idx = random.choice([0, 1, 2])
        if choice_idx == 0:
            a_count -= 1 if a_count > 1 else -1
        elif choice_idx == 1:
            b_count -= 1 if b_count > 1 else -1
        else:
            c_count -= 1 if c_count > 1 else -1

    return ("a" * a_count, "b" * b_count, "c" * c_count)
def generate_language7_positive():
    """
    Generate a string where count('a') == sum(count(other chars)),
    and 'a's are randomly distributed throughout the sentence.
    """
    other_total = random.randint(1, 10)
    others = ['b', 'c', 'd', 'e']
    counts = [0] * len(others)

    for _ in range(other_total):
        idx = random.randint(0, len(others) - 1)
        counts[idx] += 1

    a_part = ['a'] * other_total
    others_part = []
    for char, count in zip(others, counts):
        others_part.extend([char] * count)

    full_chars = a_part + others_part
    random.shuffle(full_chars)
    return ("".join(full_chars),)


def generate_language7_negative():
    """
    Generate a string where count('a') != sum(count(other chars)),
    and all characters randomly shuffled.
    """
    a_count = random.randint(1, 10)
    other_total = random.choice([i for i in range(1, 11) if i != a_count])

    others = ['b', 'c', 'd', 'e']
    counts = [0] * len(others)

    for _ in range(other_total):
        idx = random.randint(0, len(others) - 1)
        counts[idx] += 1

    a_part = ['a'] * a_count
    others_part = []
    for char, count in zip(others, counts):
        others_part.extend([char] * count)

    full_chars = a_part + others_part
    random.shuffle(full_chars)
    return ("".join(full_chars),)

def generate_language8_positive():
    sentence_length = random.randint(10, 20)
    chars = random.choices(string.ascii_lowercase, k=sentence_length)
    freq = {ch: chars.count(ch) for ch in set(chars)}
    most_common = max(freq, key=freq.get)
    value = freq[most_common] + ord(most_common)
    return f"{value}{''.join(chars)}"

def generate_language8_negative():
    sentence_length = random.randint(10, 20)
    chars = random.choices(string.ascii_lowercase, k=sentence_length)
    freq = {ch: chars.count(ch) for ch in set(chars)}
    most_common = max(freq, key=freq.get)
    correct = freq[most_common] + ord(most_common)
    wrong_number = random.choice([i for i in range(correct - 10, correct + 10) if i != correct and i > 0])
    return f"{wrong_number}{''.join(chars)}"

def generate_language9_positive():
    """
    Pattern: start and end with the same letter (max length 100)
    """
    length = random.randint(2, 100)
    char = random.choice(string.ascii_lowercase)  # Random letter
    middle_length = length - 2
    middle_str = ''.join(random.choices(string.ascii_lowercase, k=middle_length))
    return char + middle_str + char

def generate_language9_negative():
    """
    Negative: string that doesn't start and end with the same letter
    """
    length = random.randint(2, 100)
    char = random.choice(string.ascii_lowercase)
    middle_length = length - 2
    middle_str = ''.join(random.choices(string.ascii_lowercase, k=middle_length))
    # Ensure the first and last letters are not the same
    different_char = random.choice([c for c in string.ascii_lowercase if c != char])
    return char + middle_str + different_char

def generate_language10_positive():
    """
    Pattern: even length string
    """
    length = random.choice([i for i in range(2, 101) if i % 2 == 0])  # Even length
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_language10_negative():
    """
    Negative: odd length string
    """
    length = random.choice([i for i in range(2, 101) if i % 2 != 0])  # Odd length
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def flatten(*segments):
    return "".join(segments)

def write_labeled_split(pos_samples, neg_samples, train_file, test_file, train_ratio=0.8):
    labeled = [(s, 1) for s in pos_samples] + [(s, 0) for s in neg_samples]
    random.shuffle(labeled)

    split_idx = int(len(labeled) * train_ratio)
    train_data = labeled[:split_idx]
    test_data = labeled[split_idx:]

    def write_file(path, data):
        with open(path, "w") as f:
            for seq, label in data:
                f.write(f"{seq}\t{label}\n")

    write_file(train_file, train_data)
    write_file(test_file, test_data)

def generate_and_save(language_name, pos_fn, neg_fn, num_samples=5000):
    pos_samples = [flatten(*pos_fn()) for _ in range(num_samples)]
    neg_samples = [flatten(*neg_fn()) for _ in range(num_samples)]

    os.makedirs("data", exist_ok=True)
    write_labeled_split(
        pos_samples,
        neg_samples,
        f"data/{language_name}_train.txt",
        f"data/{language_name}_test.txt"
    )
    print(f"✅ Generated {language_name}_train.txt and {language_name}_test.txt")

def main():
    generate_and_save("lang1", generate_language1_positive, generate_language1_negative)
    generate_and_save("lang2", generate_language2_positive, generate_language2_negative)
    generate_and_save("lang3", generate_language3_positive, generate_language3_negative)
    generate_and_save("lang4", generate_language4_positive, generate_language4_negative)
    generate_and_save("lang5", generate_language5_positive, generate_language5_negative)
    generate_and_save("lang6", generate_language6_positive, generate_language6_negative)
    generate_and_save("lang7", generate_language7_positive, generate_language7_negative)
    generate_and_save("lang8", generate_language8_positive, generate_language8_negative, num_samples=5000)
    generate_and_save("lang9", generate_language9_positive, generate_language9_negative)  # New language 9
    generate_and_save("lang10", generate_language10_positive, generate_language10_negative)  # New language 10


if __name__ == "__main__":
    main()