import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Additive word attention layer."""

    def __init__(self, hidden_dim=100, q_size=200):
        """Initialization parameters.
        Args:
            hidden_dim (int): Input dimension.
            q_size (int): Projection size.
        """
        super(AdditiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_size = q_size
        # V_w * hi + v_w
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.q_size, bias=True), nn.Tanh())
        # q_w * proj
        self.proj_q = nn.Linear(self.q_size, 1)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, hidden_dim]
        Returns:
            outputs, weights: [B, hidden_dim], [B, seq_len]
        """
        weights = self.proj_q(self.proj(context)).squeeze(-1)
        weights = torch.softmax(weights, dim=-1)
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights


if __name__ == "__main__":
    x = torch.tensor([[[1., 2., 3.], [1., 0., 0.]]])
    model = AdditiveAttention(3, 4)
    out, w = model(x)
    print(out.shape, w.shape)
    print(out)
