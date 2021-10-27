# -*-coding:utf-8-*-
import numpy as np
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout_p):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, self.n_layers, dropout=dropout_p, bidirectional=True)

    def forward(self, text):
        batch_size = text.shape[0]
        # [batch, seq] -> [batch, seq, embed_dim]
        embedded = self.embedding(text)
        # [batch, seq, embed_dim] -> [seq, batch, embed_dim]
        embedded = embedded.permute(1, 0, 2)
        # output -> [seq, batch, hidden_dim * D]
        # hidden -> [n_layers * D, batch, hidden_dim]
        output, hidden = self.rnn(embedded)
        # [n_layers * D, batch, hidden_dim] -> [D, n_layers, batch, hidden_dim]
        hidden = hidden.view(2, self.n_layers, batch_size, self.hidden_dim)
        # [D, n_layers, batch, hidden_dim] ->  [n_layers, batch, hidden_dim]
        hidden = hidden.mean(dim=0)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embed_dim, hidden_dim, n_layers, dropout_p):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.embed_dim)
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, self.n_layers, dropout=self.dropout_p)
        self.linear = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, text, hidden):
        # [batch] -> [batch, embed_dim]
        embedded = self.embedding(text)
        # [batch, embed_dim] -> [1, batch, embed_dim]
        embedded = embedded.unsqueeze(0)
        # output -> [1, batch, hidden_dim * D]
        # hidden -> [n_layers * D, batch, hidden_dim]
        output, hidden = self.rnn(embedded, hidden)
        # [1, batch, hidden_dim * D] -> [batch, hidden_dim * D]
        output = output[0]
        # [batch, hidden_dim * D // 2] -> [batch, output_size]
        return self.linear(output), hidden

# encoder = EncoderRNN(10, embed_dim=40, hidden_dim=80, n_layers=4, dropout_p=0.5)
# decoder = DecoderRNN(10, embed_dim=40, hidden_dim=80, n_layers=4, dropout_p=0.5)
# #
# input = torch.LongTensor([
#     [6, 7, 4, 5, 1],
#     [6, 7, 4, 5, 1],
#     [6, 7, 4, 5, 1],
# ])
#
# output, hidden = encoder(input)
# print('encoder output', output.shape)
# print('encoder hidden', hidden.shape)
#
# input = torch.LongTensor([4, 5, 5])
# output, hidden = decoder(input, hidden)
# print('decoder output', output.shape)
# print(output.argmax(1))
