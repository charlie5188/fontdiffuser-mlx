"""FontDiffuser (Yang et al., AAAI 2024) in MLX.

Same network as the official PyTorch code, layer for layer, with the same parameter names, so official and
fine-tuned PyTorch checkpoints load directly (see weights.py). Differences from the PyTorch code:

- Images and feature maps are channels-last (B, H, W, C), MLX's native layout. Conv weights are stored
  (out, kh, kw, in); weights.py converts to and from PyTorch's (out, in, kh, kw).
- The deformable convolution is built from gathers and a matmul (deform.py), because MLX has no such op.
- Code paths the released model never uses (sliced attention, gradient checkpointing, FIR resampling, the
  phase-2 SCR module) are left out.

Quirks of the original that are kept on purpose, because the released weights were trained with them:
- The style encoder applies its last layer (InstanceNorm, ReLU, 1x1 conv) twice.
- The spectral-norm convolutions in both encoders recompute the weight scale from a stored vector `u0` on
  every call, and update `u0` in training mode (one power-iteration step), like BigGAN's SN.
"""
import math

import mlx.core as mx
import mlx.nn as nn

from .deform import deform_conv2d


# ------------------------------------------------------------------ small building blocks

def GroupNorm(groups, channels, eps):
    # pytorch_compatible: groups are contiguous channel ranges, as in torch.nn.GroupNorm
    return nn.GroupNorm(groups, channels, eps=eps, affine=True, pytorch_compatible=True)


def upsample_nearest(x):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


def avg_pool2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))


def instance_norm(x, eps=1e-5):
    # torch.nn.InstanceNorm2d defaults: no affine, biased variance over H and W
    mean = x.mean(axis=(1, 2), keepdims=True)
    var = ((x - mean) ** 2).mean(axis=(1, 2), keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps)


class Op(nn.Module):
    """A parameter-free layer that only holds a list slot, so parameter names line up with PyTorch's."""

    def __init__(self, fn=None):
        super().__init__()
        self._fn = fn

    def __call__(self, x):
        return self._fn(x) if self._fn else x


def timestep_embedding(t, dim, max_period=10000):
    # diffusers get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, cin, dim):
        super().__init__()
        self.linear_1 = nn.Linear(cin, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


# ------------------------------------------------------------------ attention

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=1, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim or query_dim
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = [nn.Linear(inner, query_dim)]

    def __call__(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        B, L, D = q.shape
        h = self.heads
        q = q.reshape(B, L, h, D // h).transpose(0, 2, 1, 3)
        k = k.reshape(B, k.shape[1], h, D // h).transpose(0, 2, 1, 3)
        v = v.reshape(B, v.shape[1], h, D // h).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.to_out[0](out)


class GEGLU(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.proj = nn.Linear(din, dout * 2)

    def __call__(self, x):
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)          # exact (erf) GELU, like torch.nn.functional.gelu


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = [GEGLU(dim, dim * mult), Op(), nn.Linear(dim * mult, dim)]   # net.1 was Dropout(0)

    def __call__(self, x):
        return self.net[2](self.net[0](x))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, context_dim):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def __call__(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """Style cross-attention: self-attention over positions, then attention to the style encoder's 3x3 features."""

    def __init__(self, channels, heads, dim_head, context_dim, groups=32):
        super().__init__()
        inner = heads * dim_head
        self.norm = GroupNorm(groups, channels, 1e-6)
        self.proj_in = nn.Conv2d(channels, inner, 1)
        self.transformer_blocks = [BasicTransformerBlock(inner, heads, dim_head, context_dim)]
        self.proj_out = nn.Conv2d(inner, channels, 1)

    def __call__(self, x, context):
        B, H, W, C = x.shape
        h = self.proj_in(self.norm(x))
        h = h.reshape(B, H * W, h.shape[-1])
        for blk in self.transformer_blocks:
            h = blk(h, context)
        return self.proj_out(h.reshape(B, H, W, -1)) + x


class SELayer(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.fc = [nn.Linear(channels, channels // reduction, bias=False), Op(),
                   nn.Linear(channels // reduction, channels, bias=False), Op()]

    def __call__(self, x):
        y = x.mean(axis=(1, 2))
        y = mx.sigmoid(self.fc[2](nn.silu(self.fc[0](y))))
        return x * y[:, None, None, :]


class ChannelAttnBlock(nn.Module):
    """MCA: fuse a content-encoder feature map into the UNet feature map (concat, SE attention, 1x1 down)."""

    def __init__(self, cin, cout, groups=32, eps=1e-6, reduction=32):
        super().__init__()
        self.norm1 = GroupNorm(groups, cin, eps)
        self.conv1 = nn.Conv2d(cin, cin, 1)
        self.se_channel_attn = SELayer(cin, reduction)
        self.norm3 = GroupNorm(groups, cin, eps)
        self.down_channel = nn.Conv2d(cin, cout, 1)

    def __call__(self, x, content):
        cat = mx.concatenate([x, content], axis=-1)
        h = self.conv1(nn.silu(self.norm1(cat)))
        h = self.se_channel_attn(h) + cat
        return self.down_channel(nn.silu(self.norm3(h)))


class OffsetRefStrucInter(nn.Module):
    """RSI: predicts deformable-conv offsets from the reference glyph's structure and the skip features."""

    def __init__(self, res_channels, style_channels, heads=1, groups=32):
        super().__init__()
        self.style_proj_in = nn.Conv2d(style_channels, style_channels, 1)
        self.gnorm_s = GroupNorm(groups, style_channels, 1e-6)
        self.ln_s = nn.LayerNorm(style_channels)
        self.content_proj_in = nn.Conv2d(res_channels, res_channels, 1)
        self.gnorm_c = GroupNorm(groups, res_channels, 1e-6)
        self.ln_c = nn.LayerNorm(res_channels)
        self.cross_attention = CrossAttention(style_channels, context_dim=res_channels, heads=heads, dim_head=res_channels)
        self.ff = FeedForward(style_channels)
        self.ln_ff = nn.LayerNorm(style_channels)
        self.gnorm_out = GroupNorm(groups, style_channels, 1e-6)
        self.proj_out = nn.Conv2d(style_channels, 2 * 3 * 3, 1)

    def __call__(self, res, style):
        B, H, W, Cr = res.shape
        Cs = style.shape[-1]
        s = self.ln_s(self.style_proj_in(self.gnorm_s(style)).reshape(B, H * W, Cs))
        c = self.ln_c(self.content_proj_in(self.gnorm_c(res)).reshape(B, H * W, Cr))
        h = self.cross_attention(s, context=c)
        h = self.ff(self.ln_ff(h)) + h
        return self.proj_out(self.gnorm_out(h.reshape(B, H, W, Cs)))


class DeformConv2d(nn.Module):
    """torchvision.ops.DeformConv2d(c, c, 3, padding=1): same parameters, weight stored channels-last."""

    def __init__(self, channels, k=3):
        super().__init__()
        scale = 1 / math.sqrt(channels * k * k)
        self.weight = mx.random.uniform(-scale, scale, (channels, k, k, channels))
        self.bias = mx.zeros((channels,))

    def __call__(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias, padding=1)


# ------------------------------------------------------------------ resnet / resampling

class ResnetBlock2D(nn.Module):
    def __init__(self, cin, cout, temb, eps, groups=32):
        super().__init__()
        self.norm1 = GroupNorm(groups, cin, eps)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb, cout)
        self.norm2 = GroupNorm(groups, cout, eps)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        if cin != cout:
            self.conv_shortcut = nn.Conv2d(cin, cout, 1)

    def __call__(self, x, temb):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.time_emb_proj(nn.silu(temb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        if "conv_shortcut" in self:
            x = self.conv_shortcut(x)
        return x + h


class Downsample2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def __call__(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def __call__(self, x):
        return self.conv(upsample_nearest(x))


# ------------------------------------------------------------------ UNet blocks

class DownBlock2D(nn.Module):
    def __init__(self, cin, cout, temb, eps, add_downsample, layers=2):
        super().__init__()
        self.resnets = [ResnetBlock2D(cin if i == 0 else cout, cout, temb, eps) for i in range(layers)]
        if add_downsample:
            self.downsamplers = [Downsample2D(cout)]

    def __call__(self, x, temb):
        outs = []
        for r in self.resnets:
            x = r(x, temb)
            outs.append(x)
        if "downsamplers" in self:
            x = self.downsamplers[0](x)
            outs.append(x)
        return x, outs


class MCADownBlock2D(nn.Module):
    def __init__(self, cin, cout, temb, eps, add_downsample, content_channels, context_dim, layers=2):
        super().__init__()
        self.content_attentions, self.resnets, self.style_attentions = [], [], []
        for i in range(layers):
            c = cin if i == 0 else cout
            self.content_attentions.append(ChannelAttnBlock(c + content_channels, c))
            self.resnets.append(ResnetBlock2D(c, cout, temb, eps))
            self.style_attentions.append(SpatialTransformer(cout, 1, cout, context_dim))
        if add_downsample:
            self.downsamplers = [Downsample2D(cout)]

    def __call__(self, x, temb, content, style_tokens):
        outs = []
        for ca, r, sa in zip(self.content_attentions, self.resnets, self.style_attentions):
            x = sa(r(ca(x, content), temb), style_tokens)
            outs.append(x)
        if "downsamplers" in self:
            x = self.downsamplers[0](x)
            outs.append(x)
        return x, outs


class UNetMidMCABlock2D(nn.Module):
    def __init__(self, c, temb, eps, content_channels, context_dim):
        super().__init__()
        self.resnets = [ResnetBlock2D(c, c, temb, eps), ResnetBlock2D(c, c, temb, eps)]
        self.content_attentions = [ChannelAttnBlock(c + content_channels, c)]
        self.style_attentions = [SpatialTransformer(c, 1, c, context_dim)]

    def __call__(self, x, temb, content, style_tokens):
        x = self.resnets[0](x, temb)
        x = self.content_attentions[0](x, content)
        x = self.resnets[1](x, temb)
        return self.style_attentions[0](x, style_tokens)


class UpBlock2D(nn.Module):
    def __init__(self, cin, cprev, cout, temb, eps, add_upsample, layers=3):
        super().__init__()
        self.resnets = []
        for i in range(layers):
            skip = cin if i == layers - 1 else cout
            self.resnets.append(ResnetBlock2D((cprev if i == 0 else cout) + skip, cout, temb, eps))
        if add_upsample:
            self.upsamplers = [Upsample2D(cout)]

    def __call__(self, x, skips, temb):
        for i, r in enumerate(self.resnets):
            x = r(mx.concatenate([x, skips[-1 - i]], axis=-1), temb)
        if "upsamplers" in self:
            x = self.upsamplers[0](x)
        return x


class StyleRSIUpBlock2D(nn.Module):
    def __init__(self, cin, cprev, cout, temb, eps, add_upsample, context_dim, index, layers=3):
        super().__init__()
        self.index = index
        self.sc_interpreter_offsets, self.dcn_deforms, self.resnets, self.attentions = [], [], [], []
        for i in range(layers):
            skip = cin if i == layers - 1 else cout
            self.sc_interpreter_offsets.append(OffsetRefStrucInter(skip, 128 // index))
            self.dcn_deforms.append(DeformConv2d(skip))
            self.resnets.append(ResnetBlock2D((cprev if i == 0 else cout) + skip, cout, temb, eps))
            self.attentions.append(SpatialTransformer(cout, 1, cout, context_dim))
        if add_upsample:
            self.upsamplers = [Upsample2D(cout)]

    def __call__(self, x, skips, temb, style_structure, style_tokens):
        ref = style_structure[-self.index - 2]
        total = 0.0
        for i, (off_net, dcn, r, attn) in enumerate(zip(self.sc_interpreter_offsets, self.dcn_deforms, self.resnets,
                                                        self.attentions)):
            skip = skips[-1 - i]
            offset = off_net(skip, ref)
            total = total + mx.abs(offset).mean()
            x = mx.concatenate([x, dcn(skip, offset)], axis=-1)
            x = attn(r(x, temb), style_tokens)
        if "upsamplers" in self:
            x = self.upsamplers[0](x)
        return x, total / len(self.resnets)


class UNet(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), content_start=64, context_dim=1024, eps=1e-5, ds=3):
        super().__init__()
        temb = channels[0] * 4
        self.ds = ds
        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(channels[0], temb)
        n = len(channels)
        self.down_blocks = []
        cout = channels[0]
        for i in range(n):
            cin, cout = cout, channels[i]
            last = i == n - 1
            if i in (1, 2):
                self.down_blocks.append(MCADownBlock2D(cin, cout, temb, eps, not last, content_start * 2 ** (i - 1), context_dim))
            else:
                self.down_blocks.append(DownBlock2D(cin, cout, temb, eps, not last))
        self.mid_block = UNetMidMCABlock2D(channels[-1], temb, eps, content_start * 2 ** (ds - 1), context_dim)
        rev = list(reversed(channels))
        self.up_blocks = []
        cout = rev[0]
        for i in range(n):
            cprev, cout = cout, rev[i]
            cin = rev[min(i + 1, n - 1)]
            last = i == n - 1
            if i in (1, 2):
                self.up_blocks.append(StyleRSIUpBlock2D(cin, cprev, cout, temb, eps, not last, context_dim, index=i))
            else:
                self.up_blocks.append(UpBlock2D(cin, cprev, cout, temb, eps, not last))
        self.conv_norm_out = GroupNorm(32, channels[0], eps)
        self.conv_out = nn.Conv2d(channels[0], 3, 3, padding=1)

    def __call__(self, x, t, cond):
        """x: (B, H, W, 3) noisy glyphs; t: (B,) diffusion step (may be fractional); cond: Conditions.encode output.
        Returns (noise prediction, mean absolute deformable offset), as the PyTorch UNet does."""
        temb = self.time_embedding(timestep_embedding(t, self.conv_in.weight.shape[0]))
        x = self.conv_in(x)
        skips = [x]
        for i, blk in enumerate(self.down_blocks):
            if isinstance(blk, MCADownBlock2D):
                x, outs = blk(x, temb, cond["content"][i], cond["style_tokens"])
            else:
                x, outs = blk(x, temb)
            skips += outs
        x = self.mid_block(x, temb, cond["content"][self.ds], cond["style_tokens"])
        offset = 0.0
        for blk in self.up_blocks:
            mine = skips[-len(blk.resnets):]
            skips = skips[:-len(blk.resnets)]
            if isinstance(blk, StyleRSIUpBlock2D):
                x, off = blk(x, mine, temb, cond["style_structure"], cond["style_tokens"])
                offset = offset + off
            else:
                x = blk(x, mine, temb)
        return self.conv_out(nn.silu(self.conv_norm_out(x))), offset


# ------------------------------------------------------------------ encoders (BigGAN-style, spectral norm)

class SNConv2d(nn.Module):
    """Conv with BigGAN spectral norm: weight / sigma, sigma from one power-iteration step on the stored `u0`.

    Like the PyTorch original, sigma is recomputed on every call; in training mode the new `u0` (and `sv0`, kept
    only so checkpoints round-trip) are stored back. Gradients flow through sigma but not through u and v.
    """

    def __init__(self, cin, cout, k=3, padding=1, bias=True, eps=1e-12):
        super().__init__()
        scale = 1 / math.sqrt(cin * k * k)
        self.weight = mx.random.uniform(-scale, scale, (cout, k, k, cin))
        if bias:
            self.bias = mx.zeros((cout,))
        self.u0 = mx.random.normal((1, cout))
        self.sv0 = mx.ones((1,))
        self.padding, self.eps = padding, eps
        self.freeze(keys=["u0", "sv0"], recurse=False)

    def _normalize(self, x):
        return x / mx.maximum(mx.linalg.norm(x, axis=1, keepdims=True), self.eps)

    def effective_weight(self):
        W = self.weight.reshape(self.weight.shape[0], -1)   # flattening order does not change sigma
        v = mx.stop_gradient(self._normalize(self.u0 @ W))
        u = mx.stop_gradient(self._normalize(v @ W.T))
        sigma = ((v @ W.T) @ u.T).reshape(())
        if self.training:
            self.u0 = u
            self.sv0 = mx.stop_gradient(sigma).reshape(1)
        return self.weight / sigma

    def __call__(self, x):
        y = mx.conv2d(x, self.effective_weight(), padding=self.padding)
        return y + self.bias if "bias" in self else y


class DBlock(nn.Module):
    def __init__(self, cin, cout, preactivation):
        super().__init__()
        self.preactivation = preactivation
        self.conv1 = SNConv2d(cin, cout)
        self.conv2 = SNConv2d(cout, cout)
        self.conv_sc = SNConv2d(cin, cout, k=1, padding=0)   # always learnable: every block downsamples

    def __call__(self, x):
        h = nn.relu(x) if self.preactivation else x
        h = avg_pool2(self.conv2(nn.relu(self.conv1(h))))
        sc = avg_pool2(self.conv_sc(x)) if self.preactivation else self.conv_sc(avg_pool2(x))
        return h + sc


class ContentEncoder(nn.Module):
    """96 px: 3 blocks -> features at 48 (64 ch), 24 (128 ch), 12 (256 ch)."""

    def __init__(self, ch=64):
        super().__init__()
        cins, couts = [3, ch, ch * 2], [ch, ch * 2, ch * 4]
        self.blocks = [[DBlock(i, o, preactivation=k > 0)] for k, (i, o) in enumerate(zip(cins, couts))]

    def __call__(self, x):
        feats = [x]
        for (blk,) in self.blocks:
            x = blk(x)
            feats.append(x)
        return x, feats


class StyleEncoder(nn.Module):
    """96 px: 5 blocks down to 3x3 (1024 ch), then (InstanceNorm, ReLU, 1x1 conv) applied twice, as in the original."""

    def __init__(self, ch=64):
        super().__init__()
        cins, couts = [3, ch, ch * 2, ch * 4, ch * 8], [ch, ch * 2, ch * 4, ch * 8, ch * 16]
        self.blocks = [[DBlock(i, o, preactivation=k > 0)] for k, (i, o) in enumerate(zip(cins, couts))]
        self.blocks.append([Op(instance_norm), Op(nn.relu), nn.Conv2d(couts[-1], couts[-1], 1)])

    def _last(self, x):
        for layer in self.blocks[-1]:
            x = layer(x)
        return x

    def __call__(self, x):
        feats = [x]
        for k, (blk,) in enumerate(self.blocks[:-1]):
            x = blk(x)
            if k < 4:
                feats.append(x)
        return self._last(self._last(x)), feats


# ------------------------------------------------------------------ the whole model

class FontDiffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.style_encoder = StyleEncoder()
        self.content_encoder = ContentEncoder()

    def encode(self, content, style):
        """Encoder features for a batch of (content glyph, style reference) pairs, (B, 96, 96, 3) each in [-1, 1].
        They do not depend on the diffusion step, so sampling computes them once per batch."""
        sf, _ = self.style_encoder(style)
        cf, c_feats = self.content_encoder(content)
        scf, sc_feats = self.content_encoder(style)
        B = sf.shape[0]
        return {
            "style_tokens": sf.reshape(B, -1, sf.shape[-1]),   # (B, 9, 1024)
            "content": c_feats + [cf],                          # index 1..3 used by the down and mid blocks
            "style_structure": sc_feats + [scf],                # index -3 and -4 used by the RSI up blocks
        }

    def __call__(self, x, t, content, style):
        return self.unet(x, t, self.encode(content, style))
