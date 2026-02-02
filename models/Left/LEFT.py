import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention_Blocks import AttentionLayer, ScaledDotProductAttention
from .EncDec import EncoderLayer, Encoder
from .Utils import PositionalEmbedding, PatchEmbedding
from .Tools import (TFTransform, MS_Utils, TimeEncoderSeq, FreqEncoderSeq,
                    MultiViewFusion, MsDecoder, PrototypeBank, TimeDecoder,
                    FreqDecoder, js_divergence, moving_average_1d)


class Lift(nn.Module):
    """
    Deep-fusion Lift:
      - MS multi-scale tokens (Mtok)
      - Time tokens (Ttok) from Et
      - Freq tokens (Ftok) from Ef
    fused by MultiViewFusion (cross-attn), then jointly used for:
      - MS reconstruction
      - prototype association + memory mixing
      - time/freq cycle decoding
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)

        self.tf = TFTransform(n_fft=configs.n_fft, hop_length=configs.hop_length, win_length=configs.win_length)

        with torch.no_grad():
            dummy = torch.zeros(1, self.seq_len, self.enc_in)
            _, _, _, Freq, TT = self.tf(dummy).shape
        self._Freq = int(Freq)
        self._TT = int(TT)

        ms_kernels = [int(r) for r in list(configs.ms_kernels)]
        if len(ms_kernels) == 0:
            raise ValueError("configs.ms_kernels must be non-empty, e.g. [8,4,2] (COARSE->FINE).")
        self.n_scales = len(ms_kernels)

        self.ms_utils = MS_Utils(
            factors=ms_kernels,
            boundary=configs.spectral_boundary,
            tau=configs.spectral_tau,
            eps=configs.spectral_eps,
            min_bandwidth=configs.spectral_min_bw,
        )

        self.patch_len = int(configs.patch_len)
        self.d_model = int(configs.d_model)

        patch_stride = int(getattr(configs, "patch_stride", self.patch_len))
        patch_padding = int(getattr(configs, "patch_padding", self.patch_len - 1))

        self.pos_embedding = PositionalEmbedding(self.d_model)
        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=patch_stride,
            padding=patch_padding,
            dropout=configs.patch_dropout,
        )

        e_layers = int(configs.e_layers)
        n_heads = int(configs.n_heads)
        d_ff = int(getattr(configs, "d_ff", 4 * self.d_model))

        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        ScaledDotProductAttention(attn_dropout=configs.attn_dropout),
                        d_model=self.d_model,
                        n_heads=n_heads,
                        proj_dropout=configs.proj_dropout,
                    ),
                    d_model=self.d_model,
                    d_ff=d_ff,
                    norm=configs.norm,
                    dropout=configs.ff_dropout,
                    activation=configs.activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        self.Et = TimeEncoderSeq(vs=self.enc_in, z_dim=self.d_model, d_model=self.d_model, dropout=configs.dropout)
        self.Ef = FreqEncoderSeq(vs=self.enc_in, z_dim=self.d_model, d_model=self.d_model, dropout=configs.dropout)

        fusion_layers = int(getattr(configs, "fusion_layers", 2))
        fusion_dropout = float(getattr(configs, "fusion_dropout", configs.dropout))
        fusion_d_ff = int(getattr(configs, "fusion_d_ff", d_ff))

        self.fusion = MultiViewFusion(
            d_model=self.d_model,
            n_heads=n_heads,
            d_ff=fusion_d_ff,
            dropout=fusion_dropout,
            activation=configs.activation,
            n_layers=fusion_layers
        )

        self.res_level = configs.res_level
        self.fusion_level = configs.fusion_level

        film_dropout = float(getattr(configs, "film_dropout", configs.ff_dropout))
        film_d_ff = int(getattr(configs, "film_d_ff", d_ff))
        film_activation = str(getattr(configs, "film_activation", configs.activation))

        self.ms_decoder = MsDecoder(
            d_model=self.d_model,
            patch_len=self.patch_len,
            d_ff=film_d_ff,
            dropout=film_dropout,
            activation=film_activation,
        )

        self.proto_t = PrototypeBank(num_prototypes=configs.num_prototypes, z_dim=self.d_model,
                                     temperature=configs.proto_temp, eps=configs.eps)
        self.proto_f = PrototypeBank(num_prototypes=configs.num_prototypes, z_dim=self.d_model,
                                     temperature=configs.proto_temp, eps=configs.eps)

        self.ema_momentum = float(configs.ema_momentum)
        self.conf_threshold = float(configs.conf_threshold)
        self.update_q_err = float(configs.update_q_err)
        self.update_q_ad = float(configs.update_q_ad)

        # curriculum memory mixing
        self.warmup_steps = int(configs.warmup_steps)
        self.ramp_steps = int(configs.ramp_steps)
        self.lambda_max = float(configs.lambda_max)
        self.step = 0  # only increment in training

        # Joint latent fusion (time+freq) for cycle decoders
        z_fuse_hidden = int(getattr(configs, "z_fuse_hidden", 2 * self.d_model))
        self.z_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, z_fuse_hidden),
            nn.GELU(),
            nn.Linear(z_fuse_hidden, self.d_model),
        )

        self.Dt = TimeDecoder(ts=self.seq_len, vs=self.enc_in, z_dim=self.d_model, d_model=self.d_model)
        self.Df = FreqDecoder(vs=self.enc_in, Freq=self._Freq, TT=self._TT, z_dim=self.d_model, d_model=self.d_model)

        self.alpha_ms_l = float(configs.alpha_ms_l)
        self.alpha_cycle_l = float(configs.alpha_cycle_l)
        self.alpha_cons_l = float(getattr(configs, "alpha_cons_l", 0.2))  # MS(full) <-> X_hat_from_S

        self.alpha_ms_s = float(configs.alpha_ms_s)
        self.alpha_cycle_s = float(configs.alpha_cycle_s)
        self.alpha_cons_s = float(getattr(configs, "alpha_cons_s", 0.0))

        self.alpha_freq = float(configs.alpha_freq)
        self.alpha_time = float(configs.alpha_time)
        self.alpha_gate = float(configs.alpha_gate)
        self.smooth_k = int(configs.smooth_k)
        self.eps = float(configs.eps)

        self._patch_len = self.patch_len
        self._patch_stride = patch_stride
        self._patch_padding = patch_padding
        self._ms_cache = {}

        # debug
        self.last_x_res = None

    def _get_ms_meta(self, T, device):
        """
        Compute ms_t_lens, ms_p_lens, ms_t_lens_, encoder masks, and ms_pos dynamically
        based on current input length T.
        Cache results per (T, device) to avoid recomputation.
        """
        dev_key = str(device)
        key = (int(T), dev_key)
        if key in self._ms_cache:
            return self._ms_cache[key]

        # time lengths per scale + full rate
        ms_t_lens = self.ms_utils._dummy_forward_time(int(T))  # [t1..tK, T]

        # patch token lengths for each time length (including full-rate at the end)
        patch_len = int(self._patch_len)
        stride = int(self._patch_stride)
        padding = int(self._patch_padding)

        ms_p_lens = []
        for L in ms_t_lens:
            L = int(L)
            Lpad = L + padding
            PN = (Lpad - patch_len) // stride + 1
            if PN <= 0:
                PN = 1
            ms_p_lens.append(int(PN))

        # decoder split lengths (time points after proj back to patch_len)
        ms_t_lens_ = [pn * patch_len for pn in ms_p_lens]  # length K+1

        # masks for encoder attention (encoder uses ms_p_lens)
        scale_ind_mask = self.ms_utils.scale_ind_mask(ms_p_lens).to(device)
        next_scale_mask = self.ms_utils.next_scale_mask(ms_p_lens).to(device)

        # positional embeddings for K scales only (exclude the last full-rate entry)
        ms_pos_list = [self.pos_embedding(pn) for pn in ms_p_lens[:-1]]  # K tensors [1,pn,D]
        ms_pos = self.ms_utils.concat_sampling_list(ms_pos_list).to(device)  # [1, sum(pn_k), D]

        meta = {
            "ms_t_lens": ms_t_lens,
            "ms_p_lens": ms_p_lens,
            "ms_t_lens_": ms_t_lens_,
            "scale_ind_mask": scale_ind_mask,
            "next_scale_mask": next_scale_mask,
            "ms_pos": ms_pos,
        }
        self._ms_cache[key] = meta
        return meta

    def _curriculum_lambda(self):
        if not self.training:
            return self.lambda_max
        if self.step < self.warmup_steps:
            return 0.0
        r = min(1.0, (self.step - self.warmup_steps) / max(1, self.ramp_steps))
        return self.lambda_max * r

    @staticmethod
    def _ms_weighted_smoothl1(y, x, seg_lens, weights):
        """
        y,x: (B, sum(seg_lens), C)
        seg_lens corresponds to ms_t_lens[1:] (K segments: scales 2..K and full)
        """
        err = F.smooth_l1_loss(y, x, reduction="none")  # (B,L,C)
        splits = list(torch.split(err, seg_lens, dim=1))
        if weights is None:
            weights = [1.0] * len(splits)
        w = torch.tensor(weights, device=err.device, dtype=err.dtype)
        w = w / (w.sum() + 1e-12)
        seg_means = torch.stack([s.mean() for s in splits], dim=0)  # (K,)
        return (w * seg_means).sum()

    def _forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, t, C = x_enc.shape
        if t != self.seq_len:
            raise ValueError(f"Lift expects fixed length T={self.seq_len}, got T={t}.")

        device = x_enc.device

        if self.training:
            self.step += 1

        meta = self._get_ms_meta(t, device)
        ms_t_lens = meta["ms_t_lens"]  # [t1..tK,T]
        ms_p_lens = meta["ms_p_lens"]  # [p1..pK,pT]
        ms_t_lens_ = meta["ms_t_lens_"]  # [p1*pl..pK*pl,pT*pl]
        scale_ind_mask = meta["scale_ind_mask"]
        ms_pos = meta["ms_pos"]

        # MS split (multi-scale)
        ms_x_list, x_res = self.ms_utils(x_enc, return_residual=True)  # list K of (B,ti,C), residual (B,T,C)
        self.last_x_res = x_res

        # ms_gt = concat scales 2..K + full-rate
        ms_gt = self.ms_utils.concat_sampling_list(ms_x_list[1:] + [x_enc])  # (B, sum_{i=2..K}ti + T, C)

        # patch embedding per scale (K only)
        for i in range(self.n_scales):
            xi = ms_x_list[i].permute(0, 2, 1)  # (B, C, ti)
            x_emb_i, _ = self.patch_embedding(xi)  # (B, pn_i, D)
            ms_x_list[i] = x_emb_i

        Mtok = self.ms_utils.concat_sampling_list(ms_x_list)  # (B, sum(pn_k), D)
        Mtok = Mtok + ms_pos

        # scale-independent transformer encoder (block-diag mask)
        Mtok, _ = self.encoder(Mtok, scale_ind_mask)  # (B, sum(pn_k), D)

        # Time/Freq encoders
        S = self.tf(x_enc)  # (B,C,2,F,TT)
        Ttok = self.Et(x_enc)  # (B,T,D)
        Ftok = self.Ef(S)  # (B,TT,D)

        # Deep fusion (token-level)
        Mtok_exp = Mtok.reshape(B, C, -1, self.d_model).reshape(B, -1, self.d_model)

        Ttok_f, Ftok_f, Mtok_f = self.fusion(Ttok, Ftok, Mtok_exp)

        Mtok_f = Mtok_f.reshape(B, C, -1, self.d_model).reshape(B * C, -1, self.d_model)
        # MS decode using fused Mtok_f
        #   - split K chunks, upsample to next scales incl full-rate patch tokens
        Mtok_fusion = self.res_level * Mtok + self.fusion_level * Mtok_f
        Mtok_list = self.ms_utils.split_2_list(Mtok_fusion, ms_t_lens=ms_p_lens, mode="encoder")  # K chunks
        Mtok_up_list = self.ms_utils.up(Mtok_list, ms_t_lens=ms_p_lens)  # each -> next, last -> pT
        Mtok_up = self.ms_utils.concat_sampling_list(Mtok_up_list)  # (B, sum(p2..pT), D)

        y_flat, _, _ = self.ms_decoder(Mtok_up, context=None)  # (B, sum(p2..pT)*pl, C)

        # split reconstructed sequence by ms_t_lens_ (decoder uses [1:] sizes)
        ms_x_dec = y_flat.reshape(B * C, -1, 1)  # [N, T', 1]

        # reshape back to [bs,ms_t,c] following trimming logic
        ms_x_dec = ms_x_dec.reshape(B, C, -1).permute(0, 2, 1)  # [bs, T', c]
        ms_x_dec_list = self.ms_utils.split_2_list(ms_x_dec, ms_t_lens=ms_t_lens_, mode="decoder")  # K segments

        # trim each segment to its target time length [t2..tK,T]
        for i in range(len(ms_x_dec_list)):
            ms_x_dec_list[i] = ms_x_dec_list[i][:, : ms_t_lens[i + 1], :]

        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)  # (B, ms_t, C)

        # MS loss: segment-aware smoothl1
        loss_ms = self._ms_weighted_smoothl1(
            ms_x_dec, ms_gt, seg_lens=ms_t_lens[1:], weights=getattr(self, "ms_seg_weights", None)
        )
        # MS band collapse regularizer
        loss_ms = loss_ms + float(getattr(self, "alpha_band_reg", 0.0)) * self.ms_utils.band_regularizer()

        # full-rate MS reconstruction for consistency
        X_hat_ms = ms_x_dec_list[-1]  # (B,T,C)

        # Prototype association on fused tokens
        p_t, aux_t = self.proto_t(Ttok_f)  # (B,T,M)
        p_f, aux_f = self.proto_f(Ftok_f)  # (B,TT,M)

        p_f_aligned = F.interpolate(p_f.permute(0, 2, 1), size=t, mode="linear", align_corners=False) \
            .permute(0, 2, 1).contiguous()  # (B,T,M)

        ad_t = js_divergence(p_t, p_f_aligned, eps=self.eps)  # (B,T)

        # global latents (from fused tokens)
        z_t = Ttok_f.mean(dim=1)  # (B,D)
        z_f = Ftok_f.mean(dim=1)  # (B,D)

        # prototype weights + memory read
        w_t = p_t.mean(dim=1)  # (B,M)
        w_f = p_f.mean(dim=1)  # (B,M)
        Pt = self.proto_t._norm(self.proto_t.prototypes)  # (M,D)
        Pf = self.proto_f._norm(self.proto_f.prototypes)  # (M,D)
        z_t_mem = w_t @ Pt
        z_f_mem = w_f @ Pf

        lam = self._curriculum_lambda()
        z_t_used = (1.0 - lam) * z_t + lam * z_t_mem
        z_f_used = (1.0 - lam) * z_f + lam * z_f_mem

        # joint latent for both decoders
        z_joint = self.z_fuse(torch.cat([z_t_used, z_f_used], dim=-1))  # (B,D)

        # cycle decoding
        X_hat = self.Dt(z_joint)  # (B,T,C)
        S_hat = self.Df(z_joint)  # (B,C,2,F,TT)
        X_hat_from_S = self.tf.inverse(S_hat, length=t)

        S_hat_from_X = self.tf(X_hat)
        L_cycle_tf = F.smooth_l1_loss(S_hat_from_X, S)
        L_cycle_ft = F.smooth_l1_loss(X_hat_from_S, x_enc)
        loss_cycle = L_cycle_tf + L_cycle_ft

        # consistency between MS full-rate and freq->time reconstruction
        loss_cons = F.smooth_l1_loss(X_hat_ms, X_hat_from_S.detach())

        # prototype EMA update
        if self.training:
            with torch.no_grad():
                err_b = (X_hat_from_S.detach() - x_enc).abs().mean(dim=(1, 2))  # (B,)
                ad_b = ad_t.detach().mean(dim=1)  # (B,)
                thr_err = torch.quantile(err_b, self.update_q_err)
                thr_ad = torch.quantile(ad_b, self.update_q_ad)
                conf_t_b = aux_t["max_prob"].detach().mean(dim=1)
                conf_f_b = aux_f["max_prob"].detach().mean(dim=1)
                good = (err_b <= thr_err) & (ad_b <= thr_ad) \
                       & (conf_t_b >= self.conf_threshold) & (conf_f_b >= self.conf_threshold)
                if good.any():
                    nn_t = w_t.detach()[good].argmax(dim=-1)
                    nn_f = w_f.detach()[good].argmax(dim=-1)
                    self.proto_t.ema_update_hard(z_t.detach()[good], nn_t, momentum=self.ema_momentum)
                    self.proto_f.ema_update_hard(z_f.detach()[good], nn_f, momentum=self.ema_momentum)

        cache = {
            "aux_t": aux_t,
            "aux_f": aux_f,
            "p_f_aligned": p_f_aligned,
        }
        return ms_gt, ms_x_dec, ms_t_lens, X_hat_from_S, X_hat, x_enc, loss_cycle, loss_ms, loss_cons, X_hat_ms, cache

    def _ms_anomaly_score(self, ms_x_dec, ms_gt, ms_t_lens, X_hat_from_S, X_hat, x_enc, X_hat_ms, cache):

        # MS multi-scale error aggregated to finest (full-rate) scale
        ms_err = F.smooth_l1_loss(ms_x_dec, ms_gt, reduction="none")  # (B, ms_t, C)
        ms_err_list = self.ms_utils.split_2_list(ms_err, ms_t_lens=ms_t_lens, mode="decoder")  # K segs

        for i in range(len(ms_err_list) - 1):
            e_i = ms_err_list[i].permute(0, 2, 1)  # (B,C,ti)
            up = F.interpolate(e_i, size=ms_err_list[-1].shape[1], mode="linear", align_corners=False)
            ms_err_list[-1] = ms_err_list[-1] + up.permute(0, 2, 1)

        score_ms = ms_err_list[-1]  # (B,T,C)

        score_freq = (X_hat_from_S - x_enc).abs()
        score_time = (X_hat - x_enc).abs()

        aux_t = cache["aux_t"]
        p_f_aligned = cache["p_f_aligned"]

        gate_t = 0.5 * (1.0 - aux_t["max_prob"]) + 0.5 * aux_t["entropy"]  # (B,T)
        max_prob_f = p_f_aligned.max(dim=-1).values
        ent_f = -(p_f_aligned * p_f_aligned.clamp_min(self.eps).log()).sum(dim=-1)
        ent_f = ent_f / (math.log(p_f_aligned.shape[-1] + 1e-6) if p_f_aligned.shape[-1] > 1 else 1.0)
        gate_f = 0.5 * (1.0 - max_prob_f) + 0.5 * ent_f
        gate_point = 0.5 * gate_t + 0.5 * gate_f  # (B,T)

        score_t = (self.alpha_freq * score_freq
                   + self.alpha_time * score_time
                   + self.alpha_gate * gate_point.unsqueeze(-1))

        # consistency score (MS full vs freq->time)
        if self.alpha_cons_s > 0:
            score_t = score_t + self.alpha_cons_s * (X_hat_ms - X_hat_from_S).abs()

        score_t = moving_average_1d(score_t, self.smooth_k)

        score = self.alpha_cycle_s * score_t + self.alpha_ms_s * score_ms
        return score

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ms_gt, ms_x_dec, ms_t_lens, X_hat_from_S, X_hat, x_enc, loss_cycle, loss_ms, loss_cons, X_hat_ms, cache = \
            self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        loss = self.alpha_cycle_l * loss_cycle + self.alpha_ms_l * loss_ms + self.alpha_cons_l * loss_cons
        return loss, None

    @torch.no_grad()
    def infer(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ms_gt, ms_x_dec, ms_t_lens, X_hat_from_S, X_hat, x_enc, _, _, _, X_hat_ms, cache = \
            self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        score = self._ms_anomaly_score(ms_x_dec, ms_gt, ms_t_lens, X_hat_from_S, X_hat, x_enc, X_hat_ms, cache)
        return score, None
