"""Frequency-domain enhancement components for MCFFENet's decoder.

This module contains all frequency-domain enhancement components used by
MCFFENet's decoder:

  - MDFSIModule:     Multi-Directional Frequency-Spatial Interaction module
               (FFT + learnable directional Gaussian masks → high/mid/low bands,
               then gated fusion with spatial features)
  - GateFusion: Pixel-wise gated fusion (sem + gate * freq, residual form)
  - SCModule:  Spatial Conv Module (splits features into high/low semantic
               components via dilated depthwise conv)
  - SFFModule: Selective Feature Fusion Module (channel attention-based
               fusion of high/low frequency-enhanced features)
  - MSFEB:     Multi-Scale Frequency Enhancement Block (wraps the above into
               a complete decoder block with residual connection)
  - EdgeConv:  Directional edge convolution (horizontal + vertical → merge)

All attribute names (self.xxx) are kept identical to the original AFENet
implementation for checkpoint compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    """Directional edge conv: horizontal (1,k) + vertical (k,1) → cat → 1×1 merge."""
    def __init__(self, dim, kernel_size=3, dilation=4, bias=False):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.h_conv = nn.Conv2d(dim, dim, (kernel_size, 1), padding=(pad, 0),
                                dilation=(dilation, 1), groups=dim, bias=bias)
        self.v_conv = nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, pad),
                                dilation=(1, dilation), groups=dim, bias=bias)
        self.merge = nn.Conv2d(dim * 2, dim, 1, bias=bias)

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        return self.merge(torch.cat([h, v], dim=1))


class SCModule(nn.Module):
    """Spatial Conv Module (SCModule).

    Splits spatial features into high/low semantic components using dilated
    depthwise convolutions with different receptive fields.

    Modifications:
    ① GELU after dw conv — two dw layers no longer equivalent to one linear conv
    ② BatchNorm after 1×1 conv — stabilizes training
    ③ 5×5 dw → 3×3 d=2 dw — same receptive field, params 25C → 9C
    ④ conv_spatial residual connection — guarantees gradient flow
    """
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        self.act = nn.GELU()
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=4, dilation=4, groups=dim, bias=False)
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim // 2, 1, bias=False), nn.BatchNorm2d(dim // 2))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim // 2, 1, bias=False), nn.BatchNorm2d(dim // 2))

    def forward(self, x):
        attn1 = self.act(self.conv0(x))
        attn2 = self.act(self.conv_spatial(attn1) + attn1)  # residual
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        return attn1, attn2


class GateFusion(nn.Module):
    """Pixel-wise gated fusion: sem + gate * freq (residual)."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=True),
        )

    def forward(self, freq, sem):
        gate = torch.sigmoid(self.gate(torch.cat([freq, sem], dim=1)))
        return sem + gate * freq


class MDFSIModule(nn.Module):
    """Multi-Directional Frequency-Spatial Interaction (MDFSIModule) module.

    Decomposes features into high/mid/low frequency bands via FFT with
    learnable directional Gaussian masks, then fuses them with spatial
    semantic features through GateFusion (gated fusion).

    Key components:
    - dafg_predictor: predicts 6 Gaussian parameters (σ_h, σ_w, θ for wide
      and tight bands) from pooled features
    - alpha_predictor: adaptive mid-band allocation between high and low
    - gate_high / gate_low: GateFusion for fusing freq features with spatial features
    """
    def __init__(self, dim, num_heads=4, bias=True, in_dim=3):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        rdim = self.get_reduction_dim(dim)
        self.dafg_predictor = nn.Sequential(
            nn.Conv2d(dim, rdim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(rdim, 6, 1, bias=True),
        )

        self.alpha_predictor = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1, bias=True),
        )

        self.gate_high = GateFusion(dim // 2)
        self.gate_low = GateFusion(dim // 2)
        self.conv2_1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2_2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, s_high, s_low, image, x):
        _, _, H, W = x.size()
        image = F.interpolate(image, (H, W), mode='bilinear')

        f_high, f_mid, f_low = self.fft(image)

        alpha = torch.sigmoid(self.alpha_predictor(x))
        f_high = f_high + alpha * f_mid
        f_low = f_low + (1.0 - alpha) * f_mid

        f_high = self.conv2_1(f_high)
        f_low = self.conv2_2(f_low)

        high_feature = self.gate_high(f_high, s_high)
        low_feature = self.gate_low(f_low, s_low)

        return high_feature, low_feature

    def get_reduction_dim(self, dim):
        if dim < 8:
            return max(2, dim)
        log_dim = math.log2(dim)
        return max(2, int(dim // log_dim))

    def _build_gaussian_mask(self, sigma_h, sigma_w, theta, X, Y):
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        Xp = X * cos_t + Y * sin_t
        Yp = -X * sin_t + Y * cos_t
        eps = 1e-6
        return torch.exp(
            -0.5 * (Yp.pow(2) / (sigma_h.pow(2) + eps) + Xp.pow(2) / (sigma_w.pow(2) + eps))
        )

    def fft(self, image):
        """MDFSIModule: FFT on raw image, 3-band decomposition."""
        with torch.amp.autocast('cuda', enabled=False):
            image = image.float()
            B = image.shape[0]
            img = self.conv1(image)
            _, _, H, W = img.shape

            pooled = F.adaptive_avg_pool2d(img, 1)
            params = self.dafg_predictor(pooled).squeeze(-1).squeeze(-1)

            sigma_h_wide = 0.05 + 1.45 * torch.sigmoid(params[:, 0:1])
            sigma_w_wide = 0.05 + 1.45 * torch.sigmoid(params[:, 1:2])
            theta_wide = (math.pi / 2.0) * torch.tanh(params[:, 2:3])

            sigma_h_tight = sigma_h_wide * torch.sigmoid(params[:, 3:4])
            sigma_w_tight = sigma_w_wide * torch.sigmoid(params[:, 4:5])
            theta_tight = (math.pi / 2.0) * torch.tanh(params[:, 5:6])

            sigma_h_wide = sigma_h_wide.view(B, 1, 1, 1)
            sigma_w_wide = sigma_w_wide.view(B, 1, 1, 1)
            theta_wide = theta_wide.view(B, 1, 1, 1)
            sigma_h_tight = sigma_h_tight.view(B, 1, 1, 1)
            sigma_w_tight = sigma_w_tight.view(B, 1, 1, 1)
            theta_tight = theta_tight.view(B, 1, 1, 1)

            Y, X = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=img.device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=img.device, dtype=torch.float32),
                indexing='ij',
            )
            Y = Y.unsqueeze(0).unsqueeze(0)
            X = X.unsqueeze(0).unsqueeze(0)

            g_wide = self._build_gaussian_mask(sigma_h_wide, sigma_w_wide, theta_wide, X, Y)
            g_tight = self._build_gaussian_mask(sigma_h_tight, sigma_w_tight, theta_tight, X, Y)

            low_mask = g_wide
            mid_mask = (g_tight - g_wide).clamp(min=0)
            high_mask = 1.0 - g_tight

            fft_result = torch.fft.fft2(img, norm='forward', dim=(-2, -1))
            fft_shift = torch.fft.fftshift(fft_result, dim=(-2, -1))

            fft_low = fft_shift * low_mask
            fft_mid = fft_shift * mid_mask
            fft_high = fft_shift * high_mask

            low = torch.fft.ifft2(torch.fft.ifftshift(fft_low, dim=(-2, -1)), norm='forward', dim=(-2, -1)).real
            mid = torch.fft.ifft2(torch.fft.ifftshift(fft_mid, dim=(-2, -1)), norm='forward', dim=(-2, -1)).real
            high = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)), norm='forward', dim=(-2, -1)).real

            return high, mid, low


class SFFModule(nn.Module):
    """Selective Feature Fusion Module (SFFModule).

    Fuses high/low frequency-enhanced features via channel attention
    (avg + max pooling → squeeze conv → sigmoid → weighted sum).
    """
    def __init__(self, dim):
        super().__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv3 = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, high_feature, low_feature, x):
        out = torch.cat([high_feature, low_feature], dim=1)
        avg_attn = torch.mean(out, dim=1, keepdim=True)
        max_attn, _ = torch.max(out, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        out = high_feature * sig[:, 0, :, :].unsqueeze(1) + low_feature * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv3(out)
        return x * out


class MSFEB(nn.Module):
    """Multi-Scale Frequency Enhancement Block (MSFEB).

    Integrates MDFSIModule (with GateFusion) with SCModule and SFFModule for adaptive
    spatial-frequency feature enhancement in the decoder.
    """
    def __init__(self, dim, num_heads=4, bias=True, in_dim=3):
        super(MSFEB, self).__init__()
        self.AFSIM = MDFSIModule(dim, num_heads, bias, in_dim)
        self.scm = SCModule(dim)
        self.fusion = SFFModule(dim)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()

    def forward(self, image, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        s_high, s_low = self.scm(x)
        high_feature, low_feature = self.AFSIM(s_high, s_low, image, x)
        out = self.fusion(high_feature, low_feature, x)
        result = self.proj_2(out)
        result = shortcut + result
        return result
