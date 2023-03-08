from torchtext.datasets.ag_news import AG_NEWS

train_iter, test_iter = AG_NEWS(split=("train", "test"))

