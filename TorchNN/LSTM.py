# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, config, num_embeddings, embedding_dim, padding_idx, label_size, embeddings):
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        if embeddings is not None:
            self.embedding.from_pretrained(torch.from_numpy(embeddings))
        self.dropout = nn.Dropout(config.dropout_embed)
        self.lstm = nn.LSTM(embedding_dim, config.hidden_size, num_layers=config.num_layers, bidirectional=True)
        self.linear = nn.Linear(config.hidden_size * 2, label_size)

    def forward(self, x, length, mask):
        x = self.embedding(x)
        h = pack_padded_sequence(x, length)
        h, _ = self.lstm(h)
        h, _ = pad_packed_sequence(h)
        h = self.dropout(h)
        h = torch.transpose(h, 0, 1)
        result = self.linear(h)

        # function 1 把有意义的字提取出来
        logit = torch.masked_select(result, mask.unsqueeze(2))
        logit = logit.view(-1, result.size(2))

        # function 2 无意义的替换掉，softmax之后为0
        return logit