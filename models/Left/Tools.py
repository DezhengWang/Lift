import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Utils import _get_activation


class MS_Utils(nn.Module):
    """
    Learnable constrained band edges (Nyquist-feasible) + residual coverage.
    - factors: downsample factors r_k in COARSE->FINE order, e.g. [8,4,2]
    - returns list of K downsampled sequences (time domain) + optional residual
    """

    def __init__(
            self,
            factors,
            boundary="reflect",  # "reflect" or "periodic"
            tau=0.02,  # softness for sigmoid steps
            eps=1e-8,
            min_bandwidth=0.0,  # optional collapse regularizer
    ):
        super().__init__()
        factors = [int(r) for r in factors]
        if len(factors) == 0 or any(r <= 0 for r in factors):
            raise ValueError(f"Invalid ms_kernels/factors: {factors}")
        if not all(factors[i] >= factors[i + 1] for i in range(len(factors) - 1)):
            raise ValueError(
                "ms_kernels should be COARSE->FINE, e.g. [8,4,2]. "
                f"Got: {factors}"
            )

        self.factors = factors
        self.K = len(factors)
        self.boundary = boundary
        self.tau = float(tau)
        self.eps = float(eps)
        self.min_bandwidth = float(min_bandwidth)

        cutoffs = torch.tensor([0.5 / r for r in self.factors], dtype=torch.float32)  # [K]
        order = torch.argsort(cutoffs)  # ascending cutoff
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(self.K)

        self.register_buffer("cutoffs", cutoffs)
        self.register_buffer("order", order)
        self.register_buffer("inv_order", inv_order)

        # Learnable params in SORTED order
        self.u = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))

        # debug buffers
        self.last_edges_sorted = None  # [K]
        self.last_edges = None  # [K] in original order
        self.last_masks = None  # [K,F]
        self.last_m_res = None  # [F]

    # list helpers
    def concat_sampling_list(self, xs):
        return torch.concat(xs, dim=1)

    def split_2_list(self, ms_x, ms_t_lens, mode="encoder"):
        if mode == "encoder":
            return list(torch.split(ms_x, split_size_or_sections=ms_t_lens[:-1], dim=1))
        elif mode == "decoder":
            return list(torch.split(ms_x, split_size_or_sections=ms_t_lens[1:], dim=1))
        raise ValueError(f"Unknown mode: {mode}")

    def scale_ind_mask(self, ms_t_lens):
        # mask True => block attention (cross-scale blocked)
        L = sum(t_len for t_len in ms_t_lens[:-1])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[:-1])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return (d != dT).reshape(1, 1, L, L).contiguous()

    def next_scale_mask(self, ms_t_lens):
        # mask True => block attending to "future" finer-scale tokens
        L = sum(t_len for t_len in ms_t_lens[1:])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[1:])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return (d < dT).reshape(1, 1, L, L).contiguous()

    def up(self, x_list, ms_t_lens):
        """
        Token upsample: x_list[i] -> length ms_t_lens[i+1]
        x_list[i]: [N, Li, D]
        """
        for i in range(len(ms_t_lens) - 1):
            x = x_list[i].transpose(1, 2)  # [N,D,Li]
            y = F.interpolate(x, size=ms_t_lens[i + 1], mode="nearest")
            x_list[i] = y.transpose(1, 2)
        return x_list

    @torch.no_grad()
    def _dummy_forward_time(self, T):
        ms_t_lens = [int(math.ceil(T / r)) for r in self.factors]
        ms_t_lens.append(T)
        return ms_t_lens

    def _compute_edges_sorted(self):
        """
        Build monotone edges in sorted-cutoff order:
          0 = e0 <= e1 <= ... <= eK <= cutoff_sorted[K-1]
        """
        c_sorted = self.cutoffs[self.order]  # [K]
        edges = []
        prev = torch.zeros((), device=c_sorted.device, dtype=c_sorted.dtype)

        for j in range(self.K):
            cmax = c_sorted[j]
            gap = (cmax - prev).clamp(min=0.0)
            frac = torch.sigmoid(self.u[j])
            ej = prev + gap * frac
            ej = torch.minimum(ej, cmax)
            edges.append(ej)
            prev = ej

        return torch.stack(edges, dim=0)  # [K] sorted

    def _soft_step(self, x):
        return torch.sigmoid(x / (self.tau + self.eps))

    def _build_masks(self, f):
        """
        f: [F] normalized in [0,0.5]
        Returns:
          masks_sorted: [K,F]
          m_res:        [F]
          edges_sorted: [K]
        """
        edges_sorted = self._compute_edges_sorted()
        lows = torch.cat([torch.zeros_like(edges_sorted[:1]), edges_sorted[:-1]], dim=0)
        highs = edges_sorted

        masks_sorted = []
        for j in range(self.K):
            mj = self._soft_step(f - lows[j]) - self._soft_step(f - highs[j])
            masks_sorted.append(mj.clamp_min(0.0))
        masks_sorted = torch.stack(masks_sorted, dim=0)  # [K,F]

        # residual above last edge
        last_edge = edges_sorted[-1]
        m_res_cand = self._soft_step(f - last_edge).clamp(0.0, 1.0)

        # Renormalize => coverage sum == 1
        denom = masks_sorted.sum(dim=0) + m_res_cand + self.eps
        masks_sorted = masks_sorted / denom
        m_res = m_res_cand / denom

        return masks_sorted, m_res, edges_sorted

    def down_time(self, x, return_residual=False):
        """
        x: [N,T,C] real
        outs: list length K, each [N, ceil(T/r_k), C] in ORIGINAL factor order
        x_res: [N,T,C] optional (high-frequency residual)
        """
        N, T, C = x.shape
        device, dtype = x.device, x.dtype

        if self.boundary == "reflect":
            x_ext = torch.cat([x, torch.flip(x, dims=[1])], dim=1)  # [N,2T,C]
            L = x_ext.shape[1]
        elif self.boundary == "periodic":
            x_ext = x
            L = T
        else:
            raise ValueError(self.boundary)

        X = torch.fft.rfft(x_ext, dim=1).permute(0, 2, 1)  # [N,C,F] complex
        Fbins = X.shape[-1]
        f = torch.linspace(0.0, 0.5, Fbins, device=device, dtype=torch.float32)

        masks_sorted, m_res, edges_sorted = self._build_masks(f)
        masks_orig = masks_sorted[self.inv_order]  # [K,F]
        edges_orig = edges_sorted[self.inv_order]

        with torch.no_grad():
            self.last_edges_sorted = edges_sorted.detach().clone()
            self.last_edges = edges_orig.detach().clone()
            self.last_masks = masks_orig.detach().clone()
            self.last_m_res = m_res.detach().clone()

        outs = []
        for k, r in enumerate(self.factors):
            mk = masks_orig[k].view(1, 1, -1)  # [1,1,F]
            Xk = X * mk
            xk_ext = torch.fft.irfft(Xk.permute(0, 2, 1), n=L, dim=1)  # [N,L,C]
            xk = xk_ext[:, :T, :].to(dtype=dtype)

            Tk = int(math.ceil(T / r))
            need = Tk * r
            if need > T:
                pad = need - T
                xk = torch.cat([xk, xk[:, -1:, :].expand(N, pad, C)], dim=1)

            outs.append(xk[:, ::r, :])  # [N,Tk,C]

        if not return_residual:
            return outs

        Xr = X * m_res.view(1, 1, -1)
        xr_ext = torch.fft.irfft(Xr.permute(0, 2, 1), n=L, dim=1)
        x_res = xr_ext[:, :T, :].to(dtype=dtype)
        return outs, x_res

    def band_regularizer(self):
        """
        Optional training-time regularizer to avoid band collapse.
        Penalize widths below min_bandwidth (normalized freq).
        """
        if self.min_bandwidth <= 0:
            return torch.zeros((), device=self.u.device, dtype=torch.float32)

        edges = self._compute_edges_sorted()  # [K]
        lows = torch.cat([torch.zeros_like(edges[:1]), edges[:-1]], dim=0)
        widths = edges - lows
        return F.relu(self.min_bandwidth - widths).pow(2).mean()

    def forward(self, x, return_residual=False):
        return self.down_time(x, return_residual=return_residual)


def js_divergence(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def moving_average_1d(x_btc, k):
    if k is None or k <= 1:
        return x_btc
    assert k % 2 == 1, "k must be odd"
    pad = k // 2
    x = x_btc.permute(0, 2, 1).contiguous()  # (B,C,T)
    x = F.pad(x, (pad, pad), mode="replicate")
    w = torch.ones(x.size(1), 1, k, device=x.device, dtype=x.dtype) / k
    y = F.conv1d(x, w, groups=x.size(1))
    return y.permute(0, 2, 1).contiguous()


class TFTransform(nn.Module):
    """
    Differentiable STFT/iSTFT wrapper.

    forward:
      X: (B,T,V) -> S_ri: (B,V,2,F,TT)
    inverse:
      S_ri: (B,V,2,F,TT) -> X_rec: (B,T,V)
    """

    def __init__(self, n_fft=64, hop_length=None, win_length=None, window_eps=1e-6):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else self.n_fft // 4
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        assert self.win_length <= self.n_fft
        self.window_eps = float(window_eps)
        w0 = torch.hann_window(self.win_length)
        self.window_param = nn.Parameter(w0.clone())

    def _window(self, device, dtype):
        w = F.softplus(self.window_param.to(device=device, dtype=dtype)) + self.window_eps
        w = w / (w.norm(p=2) + 1e-12) * (self.win_length ** 0.5)
        return w

    def forward(self, X):
        B, T, V = X.shape
        x = X.permute(0, 2, 1).contiguous().view(B * V, T)
        S = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self._window(x.device, x.dtype), center=True, return_complex=True
        )  # (B*V,F,TT)
        S_ri = torch.stack([S.real, S.imag], dim=1)  # (B*V,2,F,TT)
        Freq, TT = S.shape[-2], S.shape[-1]
        return S_ri.view(B, V, 2, Freq, TT)

    def inverse(self, S_ri, length):
        B, V, _, Freq, TT = S_ri.shape
        s = S_ri.view(B * V, 2, Freq, TT)
        S = torch.complex(s[:, 0], s[:, 1])
        x = torch.istft(
            S, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self._window(S.device, S.real.dtype), center=True, length=length
        )
        T = x.shape[-1]
        return x.view(B, V, T).permute(0, 2, 1).contiguous()


class TimeEncoderSeq(nn.Module):
    """X: (B,T,V) -> Ht: (B,T,D)"""

    def __init__(self, vs, d_model=128, z_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(vs, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, z_dim, kernel_size=1),
        )

    def forward(self, X):
        x = X.permute(0, 2, 1).contiguous()  # (B,V,T)
        z = self.net(x).permute(0, 2, 1).contiguous()  # (B,T,D)
        return z


class FreqEncoderSeq(nn.Module):
    """S_ri: (B,V,2,F,TT) -> Hf: (B,TT,D)"""

    def __init__(self, vs, d_model=128, z_dim=128, dropout=0.1):
        super().__init__()
        self.vs = vs
        self.net = nn.Sequential(
            nn.Conv2d(vs * 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(d_model, z_dim, kernel_size=1)

    def forward(self, S_ri):
        B, V, _, Freq, TT = S_ri.shape
        x = S_ri.view(B, V * 2, Freq, TT)  # (B,2V,F,TT)
        h = self.net(x).mean(dim=2)  # pool F -> (B,d_model,TT)
        z = self.proj(h).permute(0, 2, 1).contiguous()  # (B,TT,D)
        return z


class TimeDecoder(nn.Module):
    """z: (B,D) -> X_hat: (B,T,V)  (fixed T=ts)"""

    def __init__(self, ts, vs, z_dim=128, d_model=256):
        super().__init__()
        self.ts, self.vs = ts, vs
        self.net = nn.Sequential(
            nn.Linear(z_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, ts * vs),
        )

    def forward(self, z):
        B = z.shape[0]
        return self.net(z).view(B, self.ts, self.vs)


class FreqDecoder(nn.Module):
    """z: (B,D) -> S_hat_ri: (B,V,2,F,TT)  (fixed TT)"""

    def __init__(self, vs, Freq, TT, z_dim=128, d_model=256):
        super().__init__()
        self.vs, self.Freq, self.TT = vs, Freq, TT
        out_dim = vs * 2 * Freq * TT
        self.net = nn.Sequential(
            nn.Linear(z_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, z):
        B = z.shape[0]
        return self.net(z).view(B, self.vs, 2, self.Freq, self.TT)


class PrototypeBank(nn.Module):
    """
    q: (B,L,D) -> p: (B,L,M) + aux
    """

    def __init__(self, num_prototypes, z_dim, temperature=5.0, eps=1e-8):
        super().__init__()
        self.M = int(num_prototypes)
        self.D = int(z_dim)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.prototypes = nn.Parameter(torch.randn(self.M, self.D) * 0.02)

    def _norm(self, x):
        return x / (x.norm(p=2, dim=-1, keepdim=True) + self.eps)

    def forward(self, q):
        B, L, D = q.shape
        assert D == self.D
        qn = self._norm(q)
        pn = self._norm(self.prototypes)
        sim = torch.einsum("bld,md->blm", qn, pn)  # cosine
        p = torch.softmax(self.temperature * sim, dim=-1)

        max_prob, nn_idx = p.max(dim=-1)
        max_sim, _ = sim.max(dim=-1)
        ent = -(p * p.clamp_min(self.eps).log()).sum(dim=-1)
        ent_norm = ent / (math.log(self.M + 1e-6) if self.M > 1 else 1.0)

        aux = {"p": p, "sim": sim, "max_prob": max_prob, "max_sim": max_sim, "nn_idx": nn_idx, "entropy": ent_norm}
        return p, aux

    @torch.no_grad()
    def ema_update_hard(self, z, nn_idx, momentum=0.99):
        if z.numel() == 0:
            return
        z = self._norm(z)
        for k in nn_idx.unique():
            sel = (nn_idx == k)
            if sel.any():
                z_mean = z[sel].mean(dim=0)
                self.prototypes.data[k] = momentum * self.prototypes.data[k] + (1.0 - momentum) * z_mean


class CrossAttnFFNBlock(nn.Module):
    """
    q <- q + CrossAttn(LN(q), LN(kv))
    q <- q + FFN(LN(q))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        attn_out, _ = self.mha(
            query=qn, key=kvn, value=kvn,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            need_weights=False
        )
        q = q + self.drop(attn_out)
        q = q + self.ff(self.ln_ff(q))
        return q


class MultiViewFusion(nn.Module):
    """
    A small stack to fuse (Ttok, Ftok, Mtok) via cross-attn.

    One layer:
      T <- Attn(T, F)
      F <- Attn(F, T)
      M <- Attn(M, T)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, activation, n_layers=2):
        super().__init__()
        self.n_layers = int(n_layers)

        self.t_from_f = nn.ModuleList([
            CrossAttnFFNBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(self.n_layers)
        ])
        self.f_from_t = nn.ModuleList([
            CrossAttnFFNBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(self.n_layers)
        ])
        self.m_from_t = nn.ModuleList([
            CrossAttnFFNBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(self.n_layers)
        ])

    def forward(self, Ttok, Ftok, Mtok):
        for i in range(self.n_layers):
            Ttok = self.t_from_f[i](Ttok, Ftok)
            Ftok = self.f_from_t[i](Ftok, Ttok)
            Mtok = self.m_from_t[i](Mtok, Ttok)
        return Ttok, Ftok, Mtok


class MsDecoder(nn.Module):
    """
    Attention-free decoder with 1-token context:
      context [N, 1, D] -> gamma/beta
      x_tokens modulated then FFN then projection
    """

    def __init__(self, d_model, patch_len, d_ff, dropout, activation):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_len),
            nn.Flatten(-2),
        )

    def forward(self, x_tokens, context=None, attn_mask=None):
        if context is None:
            x = x_tokens + self.ff(x_tokens)
            y = self.proj(x)
            return y, None, None

        ctx = context.squeeze(1)  # [N,D]
        gb = self.to_gb(ctx)
        gamma, beta = gb.chunk(2, dim=-1)  # [N,D],[N,D]
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)

        x = x_tokens * (1.0 + gamma) + beta
        x = x + self.ff(x)
        y = self.proj(x)
        return y, None, None
