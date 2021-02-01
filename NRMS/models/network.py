import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.attention import AdditiveAttention


class NRMS(nn.Module):
    def __init__(self, hyperParams, weight=None):
        super(NRMS, self).__init__()
        self.hyperParams = hyperParams
        self.news_encoder = Encoder(hyperParams, weight=weight)
        self.multi_head = nn.MultiheadAttention(hyperParams["hidden_size"], hyperParams["head_num"], dropout=0.1)
        self.projection = nn.Linear(hyperParams["hidden_size"], hyperParams["hidden_size"])
        self.additive_attention = AdditiveAttention(hyperParams["hidden_size"], hyperParams["q_size"])
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, clicks, candidates, labels=None):
        """

        :param clicks: [batch_size, num_clicks, seq_len]
        :param candidates: [batch_size, num_candidates, seq_len]
        :param labels: [batch_size, num_candidates]
        :return: eval: score with activation; train: loss, score
        """
        batch_size, num_clicks, seq_len = clicks.shape[0], clicks.shape[1], clicks.shape[2]
        num_candidates = candidates.shape[1]
        clicks = clicks.reshape(-1, seq_len)
        candidates = candidates.reshape(-1, seq_len)

        # news encoder
        clicks = self.news_encoder(clicks)
        candidates = self.news_encoder(candidates)
        clicks = clicks.reshape(batch_size, num_clicks, -1)
        candidates = candidates.reshape(batch_size, num_candidates, -1)

        # multi-head attention
        clicks = clicks.permute(1, 0, 2)
        clicks, _ = self.multi_head(clicks, clicks, clicks)
        clicks = F.dropout(clicks.permute(1, 0, 2), p=0.2)

        # additive attention
        clicks = self.projection(clicks)
        clicks, _ = self.additive_attention(clicks)

        # click predictor
        prediction = torch.bmm(clicks.unsqueeze(1), candidates.permute(0, 2, 1)).squeeze(1)

        # evaluation
        if labels is None:
            return torch.sigmoid(prediction)
        # compute loss
        _, labels = labels.max(dim=1)
        loss = self.lossFn(prediction, labels)
        return loss, prediction
