import torch
import torch.nn as nn
class CausalLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def phi(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.size()
        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply φ
        q_phi = self.phi(q)  # (B, nh, T, hs)
        k_phi = self.phi(k)  # (B, nh, T, hs)

        # Cumulative sums
        k_cum = torch.cumsum(k_phi, dim=2)  # (B, nh, T, hs)
        kv = k_phi.unsqueeze(-1) * v.unsqueeze(-2)  # outer product: (B, nh, T, hs, hs)
        kv_cum = torch.cumsum(kv, dim=2)  # (B, nh, T, hs, hs)

        numerator = torch.einsum("bntd,bntdh->bnth", q_phi, kv_cum)

        denominator = torch.einsum("bntd,bntd->bnt", q_phi, k_cum).unsqueeze(-1) + 1e-6

        # Output
        y = numerator / denominator  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y