import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def generate_vocab(data_iterator):
    vocab = build_vocab_from_iterator(yield_tokens(data_iterator), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def text_pipeline_generator(vocab):
    return lambda x: vocab(tokenizer(x))


def label_pipeline(x):
    return int(x) - 1
