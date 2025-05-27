import pickle
import matplotlib.pyplot as plt


def plot_dev_accuracies(filenames, labels, title, save_png=True):
    plt.figure(figsize=(10, 6))
    for file, label in zip(filenames, labels):
        with open(file, "rb") as f:
            data = pickle.load(f)
        if not data:
            print(f"Warning: {file} empty or no data")
            continue
        x, y = zip(*data)
        x = [xi / 100.0 for xi in x]  # sentences seen / 100 for x-axis
        plt.plot(x, y, label=label)
    plt.xlabel("Sentences seen / 100")
    plt.ylabel("Dev Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_png:
        png_name = title.replace(" ", "_").lower() + ".png"
        plt.savefig(png_name)
        print(f"Plot saved to {png_name}")
    plt.show()


if __name__ == "__main__":
    # Example for POS dataset
    pos_files = [
        "dev_acc_pos_a.pkl",
        "dev_acc_pos_b.pkl",
        "dev_acc_pos_c.pkl",
        "dev_acc_pos_d.pkl",
    ]
    pos_labels = ["a", "b", "c", "d"]
    plot_dev_accuracies(pos_files, pos_labels, "POS Dev Accuracy")

    # Example for NER dataset
    ner_files = [
        "dev_acc_ner_a.pkl",
        "dev_acc_ner_b.pkl",
        "dev_acc_ner_c.pkl",
        "dev_acc_ner_d.pkl",
    ]
    ner_labels = ["a", "b", "c", "d"]
    plot_dev_accuracies(ner_files, ner_labels, "NER Dev Accuracy")
