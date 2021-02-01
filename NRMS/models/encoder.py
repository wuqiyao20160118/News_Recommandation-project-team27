import torch.nn as nn
import torch.nn.functional as F
from models.attention import AdditiveAttention


class Encoder(nn.Module):
    def __init__(self, hyperParams, weight=None):
        """
        Initialization

        :param hyperParams: configurations
        :param weight: embedding weight
        """
        super(Encoder, self).__init__()
        self.hyperParams = hyperParams
        if weight is None:
            self.embedding = nn.Embedding(100, 300)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.multi_head_attention = nn.MultiheadAttention(hyperParams['embedding_size'],
                                                          num_heads=hyperParams['head_num'],
                                                          dropout=0.1)
        self.projection = nn.Linear(hyperParams['embedding_size'], hyperParams['hidden_size'])
        self.additive_attention = AdditiveAttention(hyperParams['hidden_size'], hyperParams['q_size'])

    def forward(self, x):
        """
        forward pass

        :param x: Input indexed words
        :return: encoded tensor
        """
        # embedding is like a learnable and weighted dictionary lookup
        x = F.dropout(self.embedding(x), 0.1)
        # query, key, value:`(L, N, E)` where L is the target sequence length, N is the batch size
        # , E is the embedding dimension.
        x = x.permute(1, 0, 2)
        # self-attention (multi-head)
        out, _ = self.multi_head_attention(query=x, key=x, value=x)
        out = F.dropout(out.permute(1, 0, 2))
        out = self.projection(out)
        out, _ = self.additive_attention(out)
        return out
