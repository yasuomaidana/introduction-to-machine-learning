from torch import nn


class EasyTextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(EasyTextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initial_range = 0.5
        self.embedding.weight.data.uniform_(-initial_range, initial_range)
        self.fc.weight.data.uniform_(-initial_range, initial_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
