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


def load_corpus(file_path: str) -> list[list[str]]:
    """
    Load and tokenize a corpus from a file.

    Args:
        file_path (str): Path to the corpus file.

    Returns:
        List[List[str]]: Tokenized lines from the corpus.
    """
    with open(file_path, 'rt') as corpus:
        return [line.split() for line in corpus.readlines()]


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
        full_text = load_corpus("arthur-conan-doyle.tok.train.txt")
        data = {'n': n_value, 'model': count_ngrams(full_text, n=n_value)}
        with open(path, "wb") as file_out:
            # noinspection PyTypeChecker
            pickle.dump(data, file_out)
    return data


def generate_context(context: list[str], n: int, V: set) -> tuple[str, ...]:
    """
    Generate the context for an n-gram model by processing the input context.

    Args:
        context (list[str]): The preceding tokens (n-1 context) for the current token.
        n (int): The size of the n-grams.
        V (set): The vocabulary set containing all valid tokens.

    Returns:
        tuple[str, ...]: A tuple representing the processed context, padded with <BOS> tokens
        if necessary and replacing out-of-vocabulary tokens with <OOV>.
    """

    # Take only the n-1 most recent context (Markov Assumption)
    context = tuple(context[-n + 1:])
    # Add <BOS> tokens if the context is too short, i.e., it's at the start of the sequence
    while len(context) < (n - 1):
        context = (BOS,) + context
    # Handle words that were not encountered during the training by replacing them with a special <OOV> token
    return tuple((c if c in V else OOV) for c in context)


class NGramLM:
    def __init__(self, path, smoothing=0.001, verbose=False):
        data = load_ngrams(path)
        self.n = data['n']
        self.V = set(data['V'])
        self.model = data['model']
        self.smoothing = smoothing
        self.verbose = verbose

    def get_prob(self, context, token):
        """
        Calculate the probability of a token given its context using the n-gram model.
    
        Args:
            context (list[str]): The preceding tokens (n-1 context) for the token.
            token (str): The token for which the probability is calculated.
    
        Returns:
            float: The probability of the token given the context.
        """

        # Take only the n-1 most recent context (Markov Assumption)
        context = generate_context(context, self.n, V=self.V)

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


class NGramLMMaximumLikelihood(NGramLM):
    """
    Extends the NGramLM class by implementing a Maximum Likelihood approach,
    optionally applying Laplace smoothing. Provides a method to compute the
    probability distribution for any given context.
    """

    def __init__(self, path, smoothing=0.001, verbose=False):
        super().__init__(path, smoothing, verbose)

    def get_prob_dist(self, context):
        """
        Computes the probability distribution over all valid tokens
        given the specified context using Maximum Likelihood Estimation.

        Args:
            context (list[str]): The preceding tokens acting as context.

        Returns:
            dict[str, float]: A mapping from each token to its probability,
            sorted in descending order by probability.
        """
        # Take only the n-1 most recent context (Markov Assumption)
        context = generate_context(context, self.n, V=self.V)
        if context in self.model:
            # Compute the probability distribution using a Maximum Likelihood Estimation and Laplace Smoothing
            norm = sum(self.model[context].values()) + self.smoothing * len(self.V)
            prob_dist = {k: (c + self.smoothing) / norm for k, c in self.model[context].items()}
            for word in self.V - prob_dist.keys():
                prob_dist[word] = self.smoothing / norm
        else:
            # Simplified formula if we never encountered this context; the probability of all tokens is uniform
            prob = 1 / len(self.V)
            prob_dist = {k: prob for k in self.V}
        prob_dist = dict(sorted(prob_dist.items(), key=lambda x: (-x[1], x[0])))
        return prob_dist
