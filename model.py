"""
AC-Net Model — matches EXACTLY the ACNet class from the training notebook.

Checkpoint key structure (from acnet_best.pth):
  Encoder:    enc1, enc2, enc3, enc4  (ConvBnRelu + MaxPool via Down)
  Bottleneck: bottle (Down) → vtm (VTM)
  Skip fusion: aam4, aam3, aam2, aam1  (AAM modules)
  Decoder:    up4, up3, up2, up1  (Up modules)
  Output:     head (Conv2d 32→3)

Training config:
  IMG_SIZE = 256
  BASE_FILTERS = 32
  N_CLASSES = 3  (0=BG, 1=Liver, 2=Tumor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ConvBnRelu (enc1 uses this directly; others via Down) ──────────────────
class ConvBnRelu(nn.Module):
    """
    Double conv block: Conv-BN-ReLU-Dropout-Conv-BN-ReLU
    Checkpoint stores as .block.0, .block.1, .block.4, .block.5
    (indices 2,3 are ReLU + Dropout2d which have no weights)
    """
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # .block.0
            nn.BatchNorm2d(out_ch),                               # .block.1
            nn.ReLU(inplace=True),                                # .block.2
            nn.Dropout2d(dropout),                                # .block.3
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # .block.4
            nn.BatchNorm2d(out_ch),                               # .block.5
            nn.ReLU(inplace=True),                                # .block.6
        )

    def forward(self, x):
        return self.block(x)


# ── Down (enc2/enc3/enc4/bottle use this) ──────────────────────────────────
class Down(nn.Module):
    """
    MaxPool2d then ConvBnRelu.
    Checkpoint: .block.0 = MaxPool (no weights), .block.1 = ConvBnRelu
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),         # .block.0
            ConvBnRelu(in_ch, out_ch)  # .block.1
        )

    def forward(self, x):
        return self.block(x)


# ── Up (decoder blocks) ────────────────────────────────────────────────────
class Up(nn.Module):
    """
    ConvTranspose2d upsample, cat with AAM-processed skip, then ConvBnRelu.
    Checkpoint: .up = ConvTranspose2d, .conv = ConvBnRelu
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── ChannelNorm: channel-wise affine [C,1,1] ──────────────────────────────
class ChannelNorm(nn.Module):
    """
    Learnable per-channel scale+shift stored as (C,1,1) tensors.
    Checkpoint keys: .norm1.weight [C,1,1], .norm1.bias [C,1,1]
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


# ── AAM: Axial Attention Module ────────────────────────────────────────────
class AAM(nn.Module):
    """
    Encoder + Decoder feature fusion with H-axis and W-axis attention.
    Checkpoint keys: .align, .fuse, .h_attn, .w_attn, .norm1, .norm2
    Note: norm1/norm2 have shape [C, 1, 1] (not standard LayerNorm shape)
    """
    def __init__(self, enc_ch, dec_ch, heads=4):
        super().__init__()
        out_ch = enc_ch
        self.align  = nn.Conv2d(dec_ch, enc_ch, 1)
        self.fuse   = ConvBnRelu(enc_ch * 2, out_ch)
        self.h_attn = nn.MultiheadAttention(out_ch, heads, batch_first=True)
        self.w_attn = nn.MultiheadAttention(out_ch, heads, batch_first=True)
        # Channel-wise norm stored as [C,1,1] weight+bias in checkpoint
        self.norm1  = ChannelNorm(out_ch)
        self.norm2  = ChannelNorm(out_ch)

    def forward(self, enc, dec):
        dec_up = F.interpolate(self.align(dec), size=enc.shape[2:],
                               mode='bilinear', align_corners=False)
        x = torch.cat([enc, dec_up], dim=1)
        x = self.fuse(x)

        B, C, H, W = x.shape

        # H-axis attention (process rows)
        xh = x.permute(0, 1, 3, 2).reshape(B * W, C, H)
        xh_seq = xh.permute(0, 2, 1)                        # (B*W, H, C)
        attn_h, _ = self.h_attn(xh_seq, xh_seq, xh_seq)
        xh = (xh_seq + attn_h).permute(0, 2, 1)             # (B*W, C, H)
        xh = xh.reshape(B, W, C, H).permute(0, 2, 3, 1)    # (B, C, H, W)

        # W-axis attention (process columns)
        xw = xh.reshape(B * H, C, W)
        xw_seq = xw.permute(0, 2, 1)                        # (B*H, W, C)
        attn_w, _ = self.w_attn(xw_seq, xw_seq, xw_seq)
        xw = (xw_seq + attn_w).permute(0, 2, 1)             # (B*H, C, W)
        xw = xw.reshape(B, H, C, W).permute(0, 2, 1, 3)    # (B, C, H, W)

        return xw


# ── VTM: Vision Transformer Module ────────────────────────────────────────
class VTM(nn.Module):
    """
    Bottleneck transformer. Flatten spatial → self-attention → reshape back.
    Checkpoint keys: .norm1, .norm2, .attn, .mlp.0, .mlp.3
    """
    def __init__(self, channels, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads,
                                           dropout=dropout, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),  # .mlp.0
            nn.GELU(),                                   # .mlp.1
            nn.Dropout(dropout),                         # .mlp.2
            nn.Linear(channels * mlp_ratio, channels),  # .mlp.3
            nn.Dropout(dropout),                         # .mlp.4
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


# ── Full ACNet ─────────────────────────────────────────────────────────────
class ACNet(nn.Module):
    """
    Exact architecture matching acnet_best.pth checkpoint.

    Input:  (B, 1, 256, 256)
    Output: (B, 3, 256, 256)  — logits for [Background, Liver, Tumor]

    Layer names match checkpoint keys exactly:
      enc1 (ConvBnRelu), enc2/enc3/enc4 (Down), bottle (Down)
      vtm (VTM)
      aam4/aam3/aam2/aam1 (AAM)
      up4/up3/up2/up1 (Up)
      head (Conv2d)
    """
    def __init__(self, in_ch=1, n_classes=3, base=32):
        super().__init__()
        f = base

        # Encoder
        self.enc1   = ConvBnRelu(in_ch, f)       # enc1.block.*    (32ch)
        self.enc2   = Down(f,     f * 2)          # enc2.block.*    (64ch)
        self.enc3   = Down(f * 2, f * 4)          # enc3.block.*    (128ch)
        self.enc4   = Down(f * 4, f * 8)          # enc4.block.*    (256ch)

        # Bottleneck
        self.bottle = Down(f * 8, f * 16)         # bottle.block.*  (512ch)
        self.vtm    = VTM(f * 16, num_heads=8)    # vtm.*

        # AAM skip-connection modules
        self.aam4 = AAM(enc_ch=f * 8,  dec_ch=f * 16, heads=4)  # aam4.*
        self.aam3 = AAM(enc_ch=f * 4,  dec_ch=f * 8,  heads=4)  # aam3.*
        self.aam2 = AAM(enc_ch=f * 2,  dec_ch=f * 4,  heads=2)  # aam2.*
        self.aam1 = AAM(enc_ch=f,      dec_ch=f * 2,  heads=2)  # aam1.*

        # Decoder
        self.up4  = Up(f * 16, f * 8,  f * 8)    # up4.*
        self.up3  = Up(f * 8,  f * 4,  f * 4)    # up3.*
        self.up2  = Up(f * 4,  f * 2,  f * 2)    # up2.*
        self.up1  = Up(f * 2,  f,      f)         # up1.*

        # Output head
        self.head = nn.Conv2d(f, n_classes, 1)    # head.*

    def forward(self, x):
        # Encoder (save skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottle(e4)
        b = self.vtm(b)

        # Decoder with AAM skip connections
        d4 = self.up4(b,  self.aam4(e4, b))
        d3 = self.up3(d4, self.aam3(e3, d4))
        d2 = self.up2(d3, self.aam2(e2, d3))
        d1 = self.up1(d2, self.aam1(e1, d2))

        return self.head(d1)
