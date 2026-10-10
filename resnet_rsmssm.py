"""MCFFENet — Mamba-CNN Fusion and Frequency Enhancement Network.

block_type='ee'  → EESSM (Sobel edge + SSM + gated residual)
block_type=None  → standard RSM OSSBlock

Everything else: MCFFENet's components (DBFFB, Decoder with
MSFEB + TransformerBlock + multi-scale MDFSIModule aggregation).

PCSHE (Progressive CNN-SSM Hybrid Encoder) combines the ResNet backbone,
four PCSHEStage SSM pathways, and stage-wise DBFFB fusion.

Usage:
    python train.py -c config/vaihingen/afenet_rsmssm_ada.py
    python train.py -c config/potsdam/afenet_rsmssm_ada.py
    python train.py -c config/loveda/afenet_rsmssm_ada.py
"""

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_
from einops import rearrange
from MDFSIM import (
    EdgeConv, SCModule, GateFusion, MDFSIModule, SFFModule, MSFEB,
)
# ── RSM's SSM components (same directory) ──
from rs_mamba_ss import OSSBlock as RSM_OSSBlock
from rs_mamba_ss import PatchMerging2D


# ═══════════════════════════════════════════════════════════════════
# AFENet's non-SSM components (copied from afenet_rsmssm_net.py)
# ═══════════════════════════════════════════════════════════════════

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


# ── MDFSIModule frequency components (from afe_dafg.py) ──



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════
# Decoder (from afenet_rsmssm_net.py — multi-scale MDFSIModule aggregation)
# ═══════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    Decoder module with MSFEBs, TransformerBlocks, and multi-scale MDFSIModule
    aggregation.
    """
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 decoder=True,
                 heads=[1, 2, 4, 8],
                 num_blocks=[4, 6, 6, 8],
                 LayerNorm_type='WithBias',
                 num_classes=6,
                 ):
        super(Decoder, self).__init__()
        self.decoder = decoder
        if self.decoder:
            self.MSFEB1 = MSFEB(decode_channels * 2 ** 3, num_heads=heads[2], bias=bias)
            self.MSFEB2 = MSFEB(decode_channels * 2 ** 2, num_heads=heads[2], bias=bias)
            self.MSFEB3 = MSFEB(decode_channels * 2 ** 1, num_heads=heads[2], bias=bias)

        self.up4 = nn.Conv2d(decode_channels * 2 ** 3, decode_channels * 2 ** 2, 1)
        self.up3 = nn.Conv2d(decode_channels * 2 ** 2, decode_channels * 2 ** 1, 1)
        self.up2 = nn.Conv2d(decode_channels * 2 ** 1, decode_channels * 2 ** 0, 1)

        self.reduce_level3 = nn.Conv2d(int(decode_channels * 2 ** 3), int(decode_channels * 2 ** 2), kernel_size=1,
                                       bias=bias)
        self.reduce_level2 = nn.Conv2d(int(decode_channels * 2 ** 2), int(decode_channels * 2 ** 1), kernel_size=1,
                                       bias=bias)
        self.reduce_level1 = nn.Conv2d(int(decode_channels * 2 ** 1), int(decode_channels * 2 ** 0), kernel_size=1,
                                       bias=bias)

        self.TB1 = nn.Sequential(*[
            TransformerBlock(dim=int(decode_channels * 2 ** 2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.TB2 = nn.Sequential(*[
            TransformerBlock(dim=int(decode_channels * 2 ** 1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.TB3 = nn.Sequential(*[
            TransformerBlock(dim=int(decode_channels * 1), num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # ====== Multi-Scale MDFSIModule feature aggregation ======
        msfeb_channels = [decode_channels * 2**3,   # MSFEB1: 512
                         decode_channels * 2**2,   # MSFEB2: 256
                         decode_channels * 2**1]   # MSFEB3: 128
        self.msfeb_squeeze1 = nn.Conv2d(msfeb_channels[0], decode_channels, 1, bias=bias)
        self.msfeb_squeeze2 = nn.Conv2d(msfeb_channels[1], decode_channels, 1, bias=bias)
        self.msfeb_squeeze3 = nn.Conv2d(msfeb_channels[2], decode_channels, 1, bias=bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(decode_channels * 4, decode_channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),
        )

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))

        self.init_weight()

    def forward(self, x, res1, res2, res3, res4, h, w):
        # MSFEB1
        inp_dec_level4 = self.MSFEB1(x, res4)
        msfeb1_feat = inp_dec_level4
        inp_dec_level4 = self.up4(inp_dec_level4)
        inp_dec_level3 = F.interpolate(inp_dec_level4, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level3 = torch.cat([inp_dec_level3, res3], 1)
        inp_dec_level3 = self.reduce_level3(inp_dec_level3)
        # TB1
        inp_dec_level3 = self.TB1(inp_dec_level3)
        # MSFEB2
        out_dec_level3 = self.MSFEB2(x, inp_dec_level3)
        msfeb2_feat = out_dec_level3
        out_dec_level3 = self.up3(out_dec_level3)
        inp_dec_level2 = F.interpolate(out_dec_level3, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level2 = torch.cat([inp_dec_level2, res2], 1)
        inp_dec_level2 = self.reduce_level2(inp_dec_level2)
        # TB2
        inp_dec_level2 = self.TB2(inp_dec_level2)
        # MSFEB3
        out_dec_level2 = self.MSFEB3(x, inp_dec_level2)
        msfeb3_feat = out_dec_level2
        out_dec_level2 = self.up2(out_dec_level2)
        inp_dec_level1 = F.interpolate(out_dec_level2, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level1 = torch.cat([inp_dec_level1, res1], 1)
        inp_dec_level1 = self.reduce_level1(inp_dec_level1)
        # TB3
        out = self.TB3(inp_dec_level1)

        # ====== Multi-Scale MDFSIModule aggregation ======
        _, _, H4, W4 = out.shape
        msfeb1_up = F.interpolate(self.msfeb_squeeze1(msfeb1_feat), size=(H4, W4),
                                  mode='bilinear', align_corners=False)
        msfeb2_up = F.interpolate(self.msfeb_squeeze2(msfeb2_feat), size=(H4, W4),
                                  mode='bilinear', align_corners=False)
        msfeb3_up = F.interpolate(self.msfeb_squeeze3(msfeb3_feat), size=(H4, W4),
                                  mode='bilinear', align_corners=False)

        out = torch.cat([out, msfeb1_up, msfeb2_up, msfeb3_up], dim=1)
        out = self.fusion(out)

        out = self.segmentation_head(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ═══════════════════════════════════════════════════════════════════
# DBFFB — Dual-Branch Feature Fusion Block (from afenet_rsmssm_net.py)
# ═══════════════════════════════════════════════════════════════════

class DBFFB(nn.Module):
    """Dual-Branch Feature Fusion Block (DBFFB): Joint-input competitive gated fusion.

    ① Joint gate input — the gate sees BOTH branch outputs (ssm_output + local).
    ② Competitive gating — gate + (1-gate) = 1, keeping output amplitude stable.
    ③ Residual identity — res_feat + fused, guaranteeing gradient flow.
    ④ Spatial + channel dual-path gate.
    """
    def __init__(self, channels):
        super().__init__()
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.local_pw = nn.Conv2d(channels, channels, 1, bias=True)
        self.local_bn = nn.BatchNorm2d(channels)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, 1, 1, bias=True),
        )
        ca_hidden = max(channels // 4, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, ca_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ca_hidden, channels, 1, bias=False),
        )

    def forward(self, ssm_output, res_feat):
        local = F.gelu(self.local_bn(self.local_pw(self.local_dw(res_feat))))
        gate_input = torch.cat([ssm_output, local], dim=1)
        sg = torch.sigmoid(self.spatial_gate(gate_input))
        cg = torch.sigmoid(self.channel_gate(gate_input))
        fused = (ssm_output * sg + local * (1.0 - sg)) * cg
        return res_feat + fused


# ═══════════════════════════════════════════════════════════════════
# EESSM — Edge-Enhanced State Space Module
# ═══════════════════════════════════════════════════════════════════

class EESSM(nn.Module):
    """Edge-Enhanced State Space Module (EESSM): Sobel edge extraction + SSM + weighted fusion.

    SSM scans clean original features, then Sobel edge features are
    extracted and fused via a zero-initialized gate as residual enhancement.

    Input/output: (B, H, W, C) channel-last.
    """
    def __init__(self, hidden_dim,
                 ssm_d_state=16, ssm_ratio=2.0, drop_path=0.0, mlp_ratio=2.0,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared SSM (same param count as baseline)
        self.ssm = RSM_OSSBlock(
            hidden_dim=hidden_dim, drop_path=drop_path,
            ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
            mlp_ratio=mlp_ratio, forward_type="v2",
        )

        # Fixed Sobel filters (3×3, not learnable)
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0).repeat(hidden_dim, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0).repeat(hidden_dim, 1, 1, 1))

        # Edge refinement: 2 edge maps (Sobel-x + Sobel-y) → C
        self.edge_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
        )

        # Zero-init gate: sigmoid(-5) ≈ 0.007 → starts as pure SSM output
        self.edge_gate = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, H, W, C) channel-last
        Returns:
            (B, H, W, C) channel-last
        """
        B, H, W, C = x.shape
        x_2d = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # SSM scans clean original features
        out = self.ssm(x)  # (B, H, W, C)
        ssm_2d = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Edge extraction: fixed Sobel-x (vertical) + Sobel-y (horizontal)
        ev = F.conv2d(x_2d, self.sobel_x, padding='same', groups=C)
        eh = F.conv2d(x_2d, self.sobel_y, padding='same', groups=C)

        edge_feat = self.edge_conv(torch.cat([ev, eh], dim=1))  # (B, C, H, W)

        # Switch gate: gate=0 → pure SSM, gate=1 → pure edge-enhanced
        # Zero-init: sigmoid(-5) ≈ 0.007 → starts as ~pure SSM
        # If edge hurts (e.g. LoveDA), gate stays near 0 → no damage
        gate = torch.sigmoid(self.edge_gate)
        out_2d = (1.0 - gate) * ssm_2d + gate * (ssm_2d + edge_feat)
        return out_2d.permute(0, 2, 3, 1).contiguous()


# ═══════════════════════════════════════════════════════════════════
# PCSHEStage — SSM pathway of a Progressive CNN-SSM Hybrid Encoder stage
# ═══════════════════════════════════════════════════════════════════

class PCSHEStage(nn.Module):
    """SSM pathway of a Progressive CNN-SSM Hybrid Encoder (PCSHE) stage.

    The first stage receives shallow ResNet features; subsequent stages
    receive the preceding DBFFB-fused features. MCFFENet performs the
    corresponding ResNet feature extraction and DBFFB fusion.

    block_type='ee'  → EESSM (Sobel from feature map)
    block_type=None  → standard RSM OSSBlock
    """
    def __init__(self,
                 in_ch: int, out_ch: int, hidden_dim: int,
                 num_blocks: int = 2, ssm_d_state: int = 16,
                 ssm_ratio: float = 2.0, drop_path: float = 0.0,
                 mlp_ratio: float = 2.0, use_patch_merge: bool = True,
                 block_type: str = None, edge_pre_merge: bool = False,
                 **kwargs):
        super().__init__()
        self.use_patch_merge = use_patch_merge
        self.edge_pre_merge = edge_pre_merge and use_patch_merge and block_type == 'ee'

        if use_patch_merge:
            self.patch_merge = PatchMerging2D(dim=in_ch, out_dim=hidden_dim)
        else:
            self.channel_proj = nn.Linear(in_ch, hidden_dim)

        if block_type == 'ee':
            self.blocks = nn.ModuleList([
                EESSM(
                    hidden_dim=hidden_dim,
                    ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
                    drop_path=drop_path, mlp_ratio=mlp_ratio,
                ) for _ in range(num_blocks)])
        else:
            self.blocks = nn.ModuleList([
                RSM_OSSBlock(
                    hidden_dim=hidden_dim, drop_path=drop_path,
                    ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio, forward_type="v2",
                ) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, out_ch, 1))

    def forward(self, x: torch.Tensor, edge_prior: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.edge_pre_merge:
            # --- Sobel on FULL resolution (before PatchMerging2D) ---
            B, H, W, C = x.shape
            x_2d = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            ev = F.conv2d(x_2d, self.sobel_x, padding='same', groups=C)
            eh = F.conv2d(x_2d, self.sobel_y, padding='same', groups=C)
            edge_feat = self.edge_conv(torch.cat([ev, eh], dim=1))  # (B, hidden_dim, H, W)

            # PatchMerging2D on the input
            x = self.patch_merge(x)  # (B, H', W', hidden_dim)

            # Pool edge features to match post-merge resolution
            H_post, W_post = x.shape[1], x.shape[2]
            edge_feat = F.adaptive_avg_pool2d(edge_feat, (H_post, W_post))
            edge_feat = edge_feat.permute(0, 2, 3, 1).contiguous()  # (B, H', W', hidden_dim)

            # Standard SSM (no Sobel inside)
            for blk in self.blocks:
                x = blk(x)

            # Add edge as zero-init gated residual
            gate = torch.sigmoid(self.edge_gate)
            x = x + gate * edge_feat
        else:
            if self.use_patch_merge:
                x = self.patch_merge(x)
            else:
                x = self.channel_proj(x)
            # ee_prior: edge additive modulation BEFORE SSM (no suppression)
            if edge_prior is not None and hasattr(self, 'edge_mod'):
                ep = edge_prior.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
                mod = torch.tanh(self.edge_mod(ep))  # (B, H, W, C) in [-1,1]
                x = x + mod  # additive: boost edge channels, no suppression of others
            for blk in self.blocks:
                x = blk(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.output_proj(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# MCFFENet — Mamba-CNN Fusion and Frequency Enhancement Network
# ═══════════════════════════════════════════════════════════════════

class MCFFENet(nn.Module):
    """MCFFENet: Mamba-CNN Fusion and Frequency Enhancement Network.

    Encoder: Progressive CNN-SSM Hybrid Encoder (PCSHE)
      - ResNet18 backbone (swsl_resnet18, output_stride=32)
      - 4 PCSHEStage SSM pathways with DBFFB (SSM global + ResNet local)
      - EE-SSM: edge-enhanced SSM (Sobel + gated residual)
    Decoder:
      - MSFEB (MDFSIModule + SCModule + SFFModule) at each decoder level
      - TransformerBlock for feature refinement
      - Multi-scale MDFSIModule aggregation
      - Segmentation head
    """
    def __init__(self,
                 decode_channels=64, dropout=0.2, backbone_name='swsl_resnet18',
                 pretrained=True, num_classes=6, envss_num_blocks=2,
                 envss_num_blocks4=None, envss_d_state=16, envss_ssm_ratio=2.0,
                 envss_drop_path=0.0, envss_mlp_ratio=2.0,
                 block_type='ee', block_stages=(True, True, False, False),
                 max_full_seq_len=16384, window_size=32,
                 edge_pre_merge=False,
                 **kwargs):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, features_only=True, output_stride=32,
            out_indices=(1, 2, 3, 4), pretrained=pretrained)
        enc_ch = self.backbone.feature_info.channels()
        self.ebn1 = nn.GroupNorm(4, enc_ch[0])
        self.ebn2 = nn.GroupNorm(4, enc_ch[1])
        self.ebn3 = nn.GroupNorm(4, 256)
        self.ebn4 = nn.GroupNorm(4, 512)
        # Support per-stage block counts:
        #   int                        → (n, n, n, n)
        #   (n1, n2)                   → (n1, n1, n1, n2)  [backward compat]
        #   (n1, n2, n3, n4)           → (n1, n2, n3, n4) [full per-stage]
        if isinstance(envss_num_blocks, int):
            n1, n2, n3 = envss_num_blocks, envss_num_blocks, envss_num_blocks
        elif len(envss_num_blocks) == 4:
            n1, n2, n3, n4 = envss_num_blocks
        elif len(envss_num_blocks) == 2:
            n1, n2, n3 = envss_num_blocks[0], envss_num_blocks[0], envss_num_blocks[0]
        else:
            n1, n2, n3 = envss_num_blocks[0], envss_num_blocks[0], envss_num_blocks[0]
        if envss_num_blocks4 is not None:
            n4 = envss_num_blocks4
        elif 'n4' not in dir():
            n4 = n3

        blk_type_1 = block_type if block_stages[0] else None
        blk_type_2 = block_type if block_stages[1] else None
        blk_type_3 = block_type if block_stages[2] else None
        blk_type_4 = block_type if block_stages[3] else None

        self.encoder1 = PCSHEStage(
            in_ch=enc_ch[0], out_ch=enc_ch[0], hidden_dim=enc_ch[0],
            num_blocks=n1, ssm_d_state=envss_d_state, ssm_ratio=envss_ssm_ratio,
            drop_path=envss_drop_path, mlp_ratio=envss_mlp_ratio, use_patch_merge=False,
            block_type=blk_type_1, edge_pre_merge=edge_pre_merge,
            max_full_seq_len=max_full_seq_len, window_size=window_size)
        self.fusion1 = DBFFB(enc_ch[0])
        self.encoder2 = PCSHEStage(
            in_ch=enc_ch[0], out_ch=enc_ch[1], hidden_dim=enc_ch[1],
            num_blocks=n2, ssm_d_state=envss_d_state, ssm_ratio=envss_ssm_ratio,
            drop_path=envss_drop_path, mlp_ratio=envss_mlp_ratio, use_patch_merge=True,
            block_type=blk_type_2, edge_pre_merge=edge_pre_merge,
            max_full_seq_len=max_full_seq_len, window_size=window_size)
        self.fusion2 = DBFFB(enc_ch[1])
        self.encoder3 = PCSHEStage(
            in_ch=enc_ch[1], out_ch=256, hidden_dim=256,
            num_blocks=n3, ssm_d_state=envss_d_state, ssm_ratio=envss_ssm_ratio,
            drop_path=envss_drop_path, mlp_ratio=envss_mlp_ratio, use_patch_merge=True,
            block_type=blk_type_3, edge_pre_merge=edge_pre_merge,
            max_full_seq_len=max_full_seq_len, window_size=window_size)
        self.fusion3 = DBFFB(256)
        self.encoder4 = PCSHEStage(
            in_ch=256, out_ch=512, hidden_dim=512,
            num_blocks=n4, ssm_d_state=envss_d_state, ssm_ratio=envss_ssm_ratio,
            drop_path=envss_drop_path, mlp_ratio=envss_mlp_ratio, use_patch_merge=True,
            block_type=blk_type_4, edge_pre_merge=edge_pre_merge,
            max_full_seq_len=max_full_seq_len, window_size=window_size)
        self.fusion4 = DBFFB(512)
        self.decoder = Decoder(
            encoder_channels=(64, 128, 256, 512), decode_channels=decode_channels,
            dropout=dropout, num_classes=num_classes)
        self._init_weights()

    def _init_weights(self):
        modules = [self.encoder1, self.encoder2, self.encoder3, self.encoder4,
                   self.fusion1, self.fusion2, self.fusion3, self.fusion4]
        for m in modules:
            for sub_m in m.modules():
                if isinstance(sub_m, nn.Conv2d):
                    if not getattr(sub_m.weight, "_no_reinit", False):
                        nn.init.kaiming_normal_(sub_m.weight, mode='fan_out', nonlinearity='relu')
                    if sub_m.bias is not None and not getattr(sub_m.bias, "_no_reinit", False):
                        nn.init.constant_(sub_m.bias, 0)
                elif isinstance(sub_m, nn.Linear):
                    if not getattr(sub_m.weight, "_no_reinit", False):
                        trunc_normal_(sub_m.weight, std=.02)
                    if sub_m.bias is not None and not getattr(sub_m.bias, "_no_reinit", False):
                        nn.init.constant_(sub_m.bias, 0)
                elif isinstance(sub_m, nn.LayerNorm):
                    nn.init.constant_(sub_m.bias, 0)
                    nn.init.constant_(sub_m.weight, 1.0)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1_bb, res2_bb, res3_bb, res4_bb = self.backbone(x)

        ssm1 = self.encoder1(res1_bb)
        res1 = F.gelu(self.ebn1(self.fusion1(ssm1, res1_bb)))
        ssm2 = self.encoder2(res1)
        res2 = F.gelu(self.ebn2(self.fusion2(ssm2, res2_bb)))
        ssm3 = self.encoder3(res2)
        res3 = F.gelu(self.ebn3(self.fusion3(ssm3, res3_bb)))
        ssm4 = self.encoder4(res3)
        res4 = F.gelu(self.ebn4(self.fusion4(ssm4, res4_bb)))
        out = self.decoder(x, res1, res2, res3, res4, h, w)
        return out
