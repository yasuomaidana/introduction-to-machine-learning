import pickle

BOS = '<BOS>'
EOS = '<EOS>'
OOV = '<OOV>'


def build_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """
    Build n-grams from a list of tokens.

    Args:
        tokens (List[str]): List of tokens.
        n (int): The size of the n-grams.

    Returns:
        List[Tuple[str]]: List of n-grams as tuples.
    """
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def build_ngrams_ctrl(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """
    Build n-grams from a list of tokens, including control tokens.

    Args:
        tokens (List[str]): List of tokens.
        n (int): The size of the n-grams.

    Returns:
        List[Tuple[str]]: List of n-grams as tuples with control tokens.
    """
    # Add control tokens
    tokens = [BOS] * (n - 1) + tokens + [EOS] * (n - 1)
    return build_ngrams(tokens, n)


def count_ngrams(texts: list[list[str]], n: int) -> dict[tuple[str, ...], dict[str, int]]:
    """
    Count n-grams from a list of tokenized texts.

    Args:
        texts (List[List[str]]): List of tokenized texts.
        n (int): The size of the n-grams.

    Returns:
        Dict[Tuple[str, ...], Dict[str, int]]: Dictionary mapping n-grams to their counts.
    """
    ngram_counts = {}

    for tokens in texts:
        ngrams = build_ngrams_ctrl(tokens, n)
        for ngram in ngrams:
            last_word = ngram[-1]
            key = ngram[:-1]
            ngram_count = ngram_counts.get(key, {})
            ngram_count[last_word] = ngram_count.get(last_word, 0) + 1
            ngram_counts[key] = ngram_count

    return ngram_counts


def load_ngrams(path: str) -> dict:
    """
    Load n-grams from a file.

    Args:
        path (str): Path to the file.

    Returns:
        dict: A dictionary containing the loaded n-grams.
    """
    try:
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
    except FileNotFoundError:
        import re
        n_value = int(re.search(r'n(\d+)', path).group(1))
        # Load the corpus and tokenize it
        with open("arthur-conan-doyle.tok.train.txt", 'rb') as corpus:
            full_text = [line.decode().split() for line in corpus.readlines()]
        data = {'n': n_value, 'model': count_ngrams(full_text, n=n_value)}
        with open(path, "wb") as file_out:
            pickle.dump(data, file_out)
    return data


class NGramLM:
    def __init__(self, path, smoothing=0.001, verbose=False):
        data = load_ngrams(path)
        self.n = data['n']
        self.V = set(data['V'])
        self.model = data['model']
        self.smoothing = smoothing
        self.verbose = verbose

    def get_prob(self, context, token):
        # Take only the n-1 most recent context (Markov Assumption)
        context = tuple(context[-self.n + 1:])
        # Add <BOS> tokens if the context is too short, i.e., it's at the start of the sequence
        while len(context) < (self.n - 1):
            context = (BOS,) + context
        # Handle words that were not encountered during the training by replacing them with a special <OOV> token
        context = tuple((c if c in self.V else OOV) for c in context)
        if token not in self.V:
            token = OOV
        if context in self.model:
            # Compute the probability using a Maximum Likelihood Estimation and Laplace Smoothing
            count = self.model[context].get(token, 0)
            prob = (count + self.smoothing) / (sum(self.model[context].values()) + self.smoothing * len(self.V))
        else:
            # Simplified formula if we never encountered this context; the probability of all tokens is uniform
            prob = 1 / len(self.V)
        # Optional logging
        if self.verbose:
            print(f'{prob:.4n}', *context, '->', token)
        return prob
