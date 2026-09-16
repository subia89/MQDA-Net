"""Dual-Branch Discriminative Decoder with DFCAM (paper Sec. 3.3).

Two parallel U-Net decoder branches share the encoder skip connections:

* the binary branch predicts a foreground mask M_b (recall-oriented
  localisation prior);
* the semantic branch predicts voxel-wise probabilities M_s over
  {background, necrotic/non-enhancing core, peritumoral edema, enhancing tumor}.

At decoder stages three and four the Discriminative Feature Cross-Attention
Module (DFCAM) takes queries from the semantic features and keys/values from
the binary features (Eqs. 5-6).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, MedNeXtBlock, MedNeXtUp, conv_nd, resize_to


class PatchEmbed(nn.Module):
    """Patch embedding: point-wise projection to ``dim`` followed by average
    pooling over non-overlapping p^d patches.

    A dense p^d x C x d convolution would cost 52 M parameters for p = 16 in
    3D, so the projection is factorised into a 1x1 conv (C x d parameters)
    and parameter-free patch averaging.
    """

    def __init__(self, spatial_dims, in_ch, dim, patch_size):
        super().__init__()
        self.proj = conv_nd(spatial_dims)(in_ch, dim, 1)
        pool = nn.AvgPool3d if spatial_dims == 3 else nn.AvgPool2d
        self.pool = pool(patch_size, stride=patch_size, ceil_mode=True)

    def forward(self, x):
        y = self.pool(self.proj(x))           # (N, d, *grid)
        grid = y.shape[2:]
        return y.flatten(2).transpose(1, 2), grid  # (N, T, d)


class DFCAM(nn.Module):
    """Discriminative Feature Cross-Attention Module.

    Q = W_Q PatchEmbed(F_s), K = W_K PatchEmbed(F_b), V = W_V PatchEmbed(F_b)
    F_s' = F_s + MLP(LN(softmax(Q K^T / sqrt(d_k)) V))
    """

    def __init__(self, spatial_dims, channels, dim=256, patch_size=8, num_heads=1,
                 mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.embed_s = PatchEmbed(spatial_dims, channels, dim, patch_size)
        self.embed_b = PatchEmbed(spatial_dims, channels, dim, patch_size)
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.unproj = conv_nd(spatial_dims)(dim, channels, 1)
        self.dropout = dropout

    def _split(self, t):
        n, L, d = t.shape
        return t.view(n, L, self.num_heads, d // self.num_heads).transpose(1, 2)

    def forward(self, f_sem, f_bin):
        size = f_sem.shape[2:]
        tok_s, grid = self.embed_s(f_sem)
        tok_b, _ = self.embed_b(f_bin)
        q, k, v = self._split(self.w_q(tok_s)), self._split(self.w_k(tok_b)), self._split(self.w_v(tok_b))
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(tok_s.shape)
        upd = self.mlp(self.norm(attn))                       # (N, T, d)
        upd = upd.transpose(1, 2).reshape(f_sem.shape[0], -1, *grid)
        upd = resize_to(self.unproj(upd), size)
        return f_sem + upd


class DecoderStage(nn.Module):
    def __init__(self, spatial_dims, in_ch, out_ch, exp_r=4, kernel_size=3, n_blocks=1):
        super().__init__()
        self.up = MedNeXtUp(spatial_dims, in_ch, out_ch, exp_r, kernel_size)
        self.fuse = ConvNormAct(spatial_dims, out_ch * 2, out_ch, 1)
        self.blocks = nn.Sequential(*[MedNeXtBlock(spatial_dims, out_ch, exp_r, kernel_size)
                                      for _ in range(n_blocks)])

    def forward(self, x, skip):
        x = resize_to(self.up(x), skip.shape[2:])
        return self.blocks(self.fuse(torch.cat([x, skip], dim=1)))


class DualBranchDecoder(nn.Module):
    def __init__(self, spatial_dims, encoder_channels: Sequence[int], num_classes=4,
                 exp_r=4, kernel_size=3, n_blocks=1, dfcam_stages: Sequence[int] = (3, 4),
                 dfcam_dim=256, dfcam_patch_sizes: Sequence[int] = (8, 16), dfcam_heads=1):
        super().__init__()
        chs = list(encoder_channels)                  # [C0, C1, C2, C3, C4]
        pairs = [(chs[4], chs[3]), (chs[3], chs[2]), (chs[2], chs[1]), (chs[1], chs[0])]
        exps = [exp_r] * 4 if isinstance(exp_r, int) else list(exp_r)
        nbs = [n_blocks] * 4 if isinstance(n_blocks, int) else list(n_blocks)

        def make_branch():
            return nn.ModuleList([DecoderStage(spatial_dims, i, o, exps[j], kernel_size, nbs[j])
                                  for j, (i, o) in enumerate(pairs)])

        self.bin_stages = make_branch()
        self.sem_stages = make_branch()
        self.dfcam_stages = tuple(dfcam_stages)
        self.dfcam = nn.ModuleDict()
        for j, stage in enumerate(self.dfcam_stages):
            out_ch = pairs[stage - 1][1]
            p = dfcam_patch_sizes[min(j, len(dfcam_patch_sizes) - 1)]
            self.dfcam[str(stage)] = DFCAM(spatial_dims, out_ch, dfcam_dim, p, dfcam_heads)
        conv = conv_nd(spatial_dims)
        self.bin_head = conv(chs[0], 1, 1)
        self.sem_head = conv(chs[0], num_classes, 1)

    def forward(self, feats):
        f1, f2, f3, f4, f5 = feats
        skips = [f4, f3, f2, f1]
        hb = hs = f5
        for idx in range(4):
            stage = idx + 1
            hb = self.bin_stages[idx](hb, skips[idx])
            hs = self.sem_stages[idx](hs, skips[idx])
            if str(stage) in self.dfcam:
                hs = self.dfcam[str(stage)](hs, hb)
        return {
            "binary_logits": self.bin_head(hb),      # (N, 1, *S)
            "semantic_logits": self.sem_head(hs),    # (N, K, *S)
        }
