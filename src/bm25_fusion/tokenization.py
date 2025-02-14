def tokenize(text):
    """
    Simple tokenizer: splits text on whitespace.
    """
    return text.split()

if __name__ == "__main__":
    sample = "This is a sample sentence."
    print(tokenize(sample))