import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        bs, n_heads, q_len, d_k = q.size()
        scale = 1. / math.sqrt(d_k)
        attn_scores = torch.matmul(q, k) * scale  # attn_scores: [bs x n_heads x q_len x k_len]
        if attn_mask is not None:  # attn_mask: [q_len x k_len]
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights: [bs x n_heads x q_len x k_len]
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x q_len x d_v]

        return output.contiguous(), attn_weights


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_k=None, d_v=None, proj_dropout=0., qkv_bias=True):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.attn = attention

        self.proj = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K, V, attn_mask):
        bs, _, _ = Q.size()
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # q_s: [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)  # k_s: [bs x n_heads x d_k x k_len]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)  # v_s: [bs x n_heads x k_len x d_v]
        out, attn_weights = self.attn(q_s, k_s, v_s,
                                      attn_mask=attn_mask)  # out: [bs x n_heads x q_len x d_v], attn_weights: [bs x n_heads x q_len x k_len]
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, -1,
                                                        self.n_heads * self.d_v)  # out: [bs x q_len x n_heads * d_v]
        out = self.proj(out)  # out: [bs x q_len x d_model]
        attn_weights = attn_weights.mean(dim=1)  # attn_weights: [bs x q_len x k_len]
        return out, attn_weights
