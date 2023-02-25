import torch
from torchtext.datasets.ag_news import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import RNN, init
import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, test_iter = AG_NEWS(split=("train", "test"))
tokenizer = get_tokenizer('basic_english')  # It retrieves the tool which allows you to encode the text


def yield_tokens(datasets):  # Converts every text into an array of tokens, it returns it in an iterable way
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)


# check what happens if only left train iter
vocab = build_vocab_from_iterator(yield_tokens([train_iter, test_iter]),
                                  specials=["<unk>"])  # It generates the vocab, which represents the words into indexes
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def truncate_text(text_list: list):
    text_length = len(text_list)
    if text_length < max_words:
        return text_list + [0] * (max_words - text_length)
    else:
        return text_list[:max_words]


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        processed_text = truncate_text(processed_text)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list)
    return label_list.to(device), text_list.to(device)


def train(dataloader, optimizer, criterion):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


class TextClassificationRNNModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_class):
        super(TextClassificationRNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.rnn = RNN(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        rnn = self.rnn
        # For more layers iterate over layer
        self.rnn.all_weights[0][0] = torch.randn(
            size=(rnn.input_size, rnn.hidden_size))  # weights connecting input-hidden
        self.rnn.all_weights[0][1] = torch.randn(size=(rnn.hidden_size, rnn.hidden_size))

    def forward(self, batch_text):
        embedded = self.embedding(batch_text)
        # rand = torch.randn(1, embedded.shape[1], self.rnn.hidden_size).to(device)
        hs, h_T = self.rnn(embedded)
        return self.fc(torch.mean(hs, dim=1))


# Hyperparameters
EPOCHS = 15  # epoch
LR = 1e-3  # learning rate
BATCH_SIZE = 1028  # batch size for training

num_class = len(set([label for (label, text) in train_iter]))
max_words = 50
vocab_size = len(vocab)
embed_size = 50
hidden_size = 75

model = TextClassificationRNNModel(vocab_size, embed_size, hidden_size, num_class).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.025)
total_accu = None

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, optimizer, criterion)
    accu_val = evaluate(valid_dataloader, criterion)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
