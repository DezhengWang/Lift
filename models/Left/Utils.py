from torch import nn
import torch
import math


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


def _get_activation(name):
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d_model]

    def forward(self, t_len):
        return self.pe[:, :t_len]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _dummy_forward(self, input_lens):
        ms_p_lens = []
        for L in input_lens:
            dummy_x = torch.ones((1, 1, L))
            dummy_x = self.padding_patch_layer(dummy_x)
            dummy_x = dummy_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [1,1,PN,PL]
            ms_p_lens.append(dummy_x.shape[2])
        return ms_p_lens

    def forward(self, x):
        """
        x: [B,C,T]
        returns:
          x_emb:  [B*C, PN, D]
          x_patch:[B*C, PN, PL]
        """
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B,C,PN,PL]
        x_patch = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [B*C,PN,PL]
        x_emb = self.value_embedding(x_patch)  # [B*C,PN,D]
        return self.dropout(x_emb), x_patch
