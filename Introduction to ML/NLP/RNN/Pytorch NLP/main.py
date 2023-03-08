import torch
import time
from torch.utils.data import DataLoader
from data_loader.loader import train_iter, test_iter
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from data_loader.text_processor import generate_vocab
from models.text_classification_model import EasyTextClassificationModel
from data_loader.batch_processor import build_collate_batch
from runners.evaluate import evaluate
from runners.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generates vocab from train_dataset
num_class = len(set([label for (label, text) in train_iter]))
vocab = generate_vocab(train_iter)
vocab_size = len(vocab)
embed_size = 64

# Generates collate_batch from vocab
collate_batch = build_collate_batch(vocab)

# Generates model
model = EasyTextClassificationModel(vocab_size, embed_size, num_class).to(device)

# Hyper-parameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
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
    train(train_dataloader, model, optimizer, criterion, epoch)
    accu_val = evaluate(valid_dataloader, model, criterion)
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
