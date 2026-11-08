// FontDiffuser (Yang et al., AAAI 2024) in MLX Swift.
//
// A line-by-line translation of fdmlx/model.py. Same layers, same parameter names, same channels-last layout,
// so the .safetensors files written by the Python package (`fdmlx convert`, `fdmlx finetune`) load directly, and
// weights fine-tuned here load in Python. See fdmlx/model.py for the notes on the original's quirks that are kept
// on purpose (the style encoder's last layer applied twice; spectral norm updating `u0` in training mode).
//
// Property names follow the PyTorch parameter names (snake_case), because MLX Swift derives parameter keys
// from them. Lists that mix layer types ([Module]) keep the original list indices; `Op` fills the slots of
// parameter-free layers (activations, dropout) so the indices line up.

import Foundation
import MLX
import MLXNN

// MARK: - small building blocks

func groupNorm(_ groups: Int, _ channels: Int, _ eps: Float) -> GroupNorm {
    // pytorchCompatible: groups are contiguous channel ranges, as in torch.nn.GroupNorm
    GroupNorm(groupCount: groups, dimensions: channels, eps: eps, affine: true, pytorchCompatible: true)
}

func conv(_ cin: Int, _ cout: Int, _ k: Int, stride: Int = 1, padding: Int = 0) -> Conv2d {
    Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: .init(k), stride: .init(stride),
           padding: .init(padding))
}

func upsampleNearest(_ x: MLXArray) -> MLXArray {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return broadcast(x.reshaped(B, H, 1, W, 1, C), to: [B, H, 2, W, 2, C]).reshaped(B, H * 2, W * 2, C)
}

func avgPool2(_ x: MLXArray) -> MLXArray {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return x.reshaped(B, H / 2, 2, W / 2, 2, C).mean(axes: [2, 4])
}

func instanceNorm(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    // torch.nn.InstanceNorm2d defaults: no affine, biased variance over H and W
    let mean = x.mean(axes: [1, 2], keepDims: true)
    let variance = ((x - mean) ** 2).mean(axes: [1, 2], keepDims: true)
    return (x - mean) * rsqrt(variance + eps)
}

/// A parameter-free layer that only holds a list slot, so parameter names line up with PyTorch's.
final class Op: Module {}

func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Double = 10000) -> MLXArray {
    // diffusers get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0
    let half = dim / 2
    let freqs = exp(Float(-log(maxPeriod)) * MLXArray(0 ..< half).asType(.float32) / Float(half))
    let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return concatenated([cos(args), sin(args)], axis: -1)
}

final class TimestepEmbedding: Module, UnaryLayer {
    let linear_1: Linear
    let linear_2: Linear

    init(_ cin: Int, _ dim: Int) {
        linear_1 = Linear(cin, dim)
        linear_2 = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear_2(silu(linear_1(x))) }
}

// MARK: - attention

final class CrossAttention: Module {
    let heads: Int
    let scale: Float
    let to_q: Linear
    let to_k: Linear
    let to_v: Linear
    let to_out: [Linear]

    init(queryDim: Int, contextDim: Int? = nil, heads: Int = 1, dimHead: Int = 64) {
        let inner = heads * dimHead
        let contextDim = contextDim ?? queryDim
        self.heads = heads
        scale = Float(pow(Double(dimHead), -0.5))          // computed in double, like Python's dim_head ** -0.5
        to_q = Linear(queryDim, inner, bias: false)
        to_k = Linear(contextDim, inner, bias: false)
        to_v = Linear(contextDim, inner, bias: false)
        to_out = [Linear(inner, queryDim)]
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let context = context ?? x
        var q = to_q(x), k = to_k(context), v = to_v(context)
        let (B, L, D) = (q.dim(0), q.dim(1), q.dim(2))
        let h = heads
        q = q.reshaped(B, L, h, D / h).transposed(0, 2, 1, 3)
        k = k.reshaped(B, k.dim(1), h, D / h).transposed(0, 2, 1, 3)
        v = v.reshaped(B, v.dim(1), h, D / h).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        return to_out[0](out.transposed(0, 2, 1, 3).reshaped(B, L, D))
    }
}

final class GEGLU: Module, UnaryLayer {
    let proj: Linear

    init(_ din: Int, _ dout: Int) { proj = Linear(din, dout * 2) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(proj(x), parts: 2, axis: -1)
        return parts[0] * gelu(parts[1])                   // exact (erf) GELU, like torch.nn.functional.gelu
    }
}

final class FeedForward: Module, UnaryLayer {
    let net: [Module]                                      // [GEGLU, Dropout(0), Linear]

    init(_ dim: Int, mult: Int = 4) { net = [GEGLU(dim, dim * mult), Op(), Linear(dim * mult, dim)] }

    func callAsFunction(_ x: MLXArray) -> MLXArray { (net[2] as! Linear)((net[0] as! GEGLU)(x)) }
}

final class BasicTransformerBlock: Module {
    let attn1: CrossAttention
    let ff: FeedForward
    let attn2: CrossAttention
    let norm1: LayerNorm
    let norm2: LayerNorm
    let norm3: LayerNorm

    init(dim: Int, heads: Int, dimHead: Int, contextDim: Int) {
        attn1 = CrossAttention(queryDim: dim, heads: heads, dimHead: dimHead)
        ff = FeedForward(dim)
        attn2 = CrossAttention(queryDim: dim, contextDim: contextDim, heads: heads, dimHead: dimHead)
        norm1 = LayerNorm(dimensions: dim)
        norm2 = LayerNorm(dimensions: dim)
        norm3 = LayerNorm(dimensions: dim)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var x = attn1(norm1(x)) + x
        x = attn2(norm2(x), context: context) + x
        return ff(norm3(x)) + x
    }
}

/// Style cross-attention: self-attention over positions, then attention to the style encoder's 3x3 features.
final class SpatialTransformer: Module {
    let norm: GroupNorm
    let proj_in: Conv2d
    let transformer_blocks: [BasicTransformerBlock]
    let proj_out: Conv2d

    init(channels: Int, heads: Int, dimHead: Int, contextDim: Int, groups: Int = 32) {
        let inner = heads * dimHead
        norm = groupNorm(groups, channels, 1e-6)
        proj_in = conv(channels, inner, 1)
        transformer_blocks = [BasicTransformerBlock(dim: inner, heads: heads, dimHead: dimHead, contextDim: contextDim)]
        proj_out = conv(inner, channels, 1)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let (B, H, W) = (x.dim(0), x.dim(1), x.dim(2))
        var h = proj_in(norm(x))
        h = h.reshaped(B, H * W, h.dim(-1))
        for block in transformer_blocks { h = block(h, context: context) }
        return proj_out(h.reshaped(B, H, W, -1)) + x
    }
}

final class SELayer: Module, UnaryLayer {
    let fc: [Module]                                       // [Linear, SiLU, Linear, Sigmoid]

    init(_ channels: Int, reduction: Int) {
        fc = [Linear(channels, channels / reduction, bias: false), Op(),
              Linear(channels / reduction, channels, bias: false), Op()]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.mean(axes: [1, 2])
        y = sigmoid((fc[2] as! Linear)(silu((fc[0] as! Linear)(y))))
        return x * y.expandedDimensions(axes: [1, 2])
    }
}

/// MCA: fuse a content-encoder feature map into the UNet feature map (concat, SE attention, 1x1 down).
final class ChannelAttnBlock: Module {
    let norm1: GroupNorm
    let conv1: Conv2d
    let se_channel_attn: SELayer
    let norm3: GroupNorm
    let down_channel: Conv2d

    init(_ cin: Int, _ cout: Int, groups: Int = 32, eps: Float = 1e-6, reduction: Int = 32) {
        norm1 = groupNorm(groups, cin, eps)
        conv1 = conv(cin, cin, 1)
        se_channel_attn = SELayer(cin, reduction: reduction)
        norm3 = groupNorm(groups, cin, eps)
        down_channel = conv(cin, cout, 1)
    }

    func callAsFunction(_ x: MLXArray, content: MLXArray) -> MLXArray {
        let cat = concatenated([x, content], axis: -1)
        var h = conv1(silu(norm1(cat)))
        h = se_channel_attn(h) + cat
        return down_channel(silu(norm3(h)))
    }
}

/// RSI: predicts deformable-conv offsets from the reference glyph's structure and the skip features.
final class OffsetRefStrucInter: Module {
    let style_proj_in: Conv2d
    let gnorm_s: GroupNorm
    let ln_s: LayerNorm
    let content_proj_in: Conv2d
    let gnorm_c: GroupNorm
    let ln_c: LayerNorm
    let cross_attention: CrossAttention
    let ff: FeedForward
    let ln_ff: LayerNorm
    let gnorm_out: GroupNorm
    let proj_out: Conv2d

    init(resChannels: Int, styleChannels: Int, heads: Int = 1, groups: Int = 32) {
        style_proj_in = conv(styleChannels, styleChannels, 1)
        gnorm_s = groupNorm(groups, styleChannels, 1e-6)
        ln_s = LayerNorm(dimensions: styleChannels)
        content_proj_in = conv(resChannels, resChannels, 1)
        gnorm_c = groupNorm(groups, resChannels, 1e-6)
        ln_c = LayerNorm(dimensions: resChannels)
        cross_attention = CrossAttention(queryDim: styleChannels, contextDim: resChannels, heads: heads,
                                         dimHead: resChannels)
        ff = FeedForward(styleChannels)
        ln_ff = LayerNorm(dimensions: styleChannels)
        gnorm_out = groupNorm(groups, styleChannels, 1e-6)
        proj_out = conv(styleChannels, 2 * 3 * 3, 1)
    }

    func callAsFunction(_ res: MLXArray, style: MLXArray) -> MLXArray {
        let (B, H, W, Cr) = (res.dim(0), res.dim(1), res.dim(2), res.dim(3))
        let Cs = style.dim(-1)
        let s = ln_s(style_proj_in(gnorm_s(style)).reshaped(B, H * W, Cs))
        let c = ln_c(content_proj_in(gnorm_c(res)).reshaped(B, H * W, Cr))
        var h = cross_attention(s, context: c)
        h = ff(ln_ff(h)) + h
        return proj_out(gnorm_out(h.reshaped(B, H, W, Cs)))
    }
}

/// torchvision.ops.DeformConv2d(c, c, 3, padding=1): same parameters, weight stored channels-last.
final class DeformConv2d: Module {
    let weight: MLXArray
    let bias: MLXArray

    init(_ channels: Int, k: Int = 3) {
        let scale = 1 / sqrt(Float(channels * k * k))
        weight = MLXRandom.uniform(low: -scale, high: scale, [channels, k, k, channels])
        bias = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        deformConv2d(x, offset: offset, weight: weight, bias: bias, padding: 1)
    }
}

// MARK: - resnet / resampling

final class ResnetBlock2D: Module {
    let norm1: GroupNorm
    let conv1: Conv2d
    let time_emb_proj: Linear
    let norm2: GroupNorm
    let conv2: Conv2d
    let conv_shortcut: Conv2d?

    init(_ cin: Int, _ cout: Int, temb: Int, eps: Float, groups: Int = 32) {
        norm1 = groupNorm(groups, cin, eps)
        conv1 = conv(cin, cout, 3, padding: 1)
        time_emb_proj = Linear(temb, cout)
        norm2 = groupNorm(groups, cout, eps)
        conv2 = conv(cout, cout, 3, padding: 1)
        conv_shortcut = cin != cout ? conv(cin, cout, 1) : nil
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = h + time_emb_proj(silu(temb)).expandedDimensions(axes: [1, 2])
        h = conv2(silu(norm2(h)))
        return (conv_shortcut.map { $0(x) } ?? x) + h
    }
}

final class Downsample2D: Module, UnaryLayer {
    let conv: Conv2d
    init(_ c: Int) { conv = FontDiffuserMLX.conv(c, c, 3, stride: 2, padding: 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class Upsample2D: Module, UnaryLayer {
    let conv: Conv2d
    init(_ c: Int) { conv = FontDiffuserMLX.conv(c, c, 3, padding: 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(upsampleNearest(x)) }
}

// MARK: - UNet blocks

final class DownBlock2D: Module {
    let resnets: [ResnetBlock2D]
    let downsamplers: [Downsample2D]

    init(_ cin: Int, _ cout: Int, temb: Int, eps: Float, addDownsample: Bool, layers: Int = 2) {
        resnets = (0 ..< layers).map { ResnetBlock2D($0 == 0 ? cin : cout, cout, temb: temb, eps: eps) }
        downsamplers = addDownsample ? [Downsample2D(cout)] : []
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = x
        var outs = [MLXArray]()
        for r in resnets {
            x = r(x, temb: temb)
            outs.append(x)
        }
        if let d = downsamplers.first {
            x = d(x)
            outs.append(x)
        }
        return (x, outs)
    }
}

final class MCADownBlock2D: Module {
    let content_attentions: [ChannelAttnBlock]
    let resnets: [ResnetBlock2D]
    let style_attentions: [SpatialTransformer]
    let downsamplers: [Downsample2D]

    init(_ cin: Int, _ cout: Int, temb: Int, eps: Float, addDownsample: Bool, contentChannels: Int,
         contextDim: Int, layers: Int = 2) {
        var ca = [ChannelAttnBlock](), rs = [ResnetBlock2D](), sa = [SpatialTransformer]()
        for i in 0 ..< layers {
            let c = i == 0 ? cin : cout
            ca.append(ChannelAttnBlock(c + contentChannels, c))
            rs.append(ResnetBlock2D(c, cout, temb: temb, eps: eps))
            sa.append(SpatialTransformer(channels: cout, heads: 1, dimHead: cout, contextDim: contextDim))
        }
        content_attentions = ca
        resnets = rs
        style_attentions = sa
        downsamplers = addDownsample ? [Downsample2D(cout)] : []
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray, content: MLXArray, styleTokens: MLXArray)
        -> (MLXArray, [MLXArray])
    {
        var x = x
        var outs = [MLXArray]()
        for i in resnets.indices {
            x = style_attentions[i](resnets[i](content_attentions[i](x, content: content), temb: temb),
                                    context: styleTokens)
            outs.append(x)
        }
        if let d = downsamplers.first {
            x = d(x)
            outs.append(x)
        }
        return (x, outs)
    }
}

final class UNetMidMCABlock2D: Module {
    let resnets: [ResnetBlock2D]
    let content_attentions: [ChannelAttnBlock]
    let style_attentions: [SpatialTransformer]

    init(_ c: Int, temb: Int, eps: Float, contentChannels: Int, contextDim: Int) {
        resnets = [ResnetBlock2D(c, c, temb: temb, eps: eps), ResnetBlock2D(c, c, temb: temb, eps: eps)]
        content_attentions = [ChannelAttnBlock(c + contentChannels, c)]
        style_attentions = [SpatialTransformer(channels: c, heads: 1, dimHead: c, contextDim: contextDim)]
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray, content: MLXArray, styleTokens: MLXArray) -> MLXArray {
        var x = resnets[0](x, temb: temb)
        x = content_attentions[0](x, content: content)
        x = resnets[1](x, temb: temb)
        return style_attentions[0](x, context: styleTokens)
    }
}

final class UpBlock2D: Module {
    let resnets: [ResnetBlock2D]
    let upsamplers: [Upsample2D]

    init(_ cin: Int, _ cprev: Int, _ cout: Int, temb: Int, eps: Float, addUpsample: Bool, layers: Int = 3) {
        resnets = (0 ..< layers).map { i in
            let skip = i == layers - 1 ? cin : cout
            return ResnetBlock2D((i == 0 ? cprev : cout) + skip, cout, temb: temb, eps: eps)
        }
        upsamplers = addUpsample ? [Upsample2D(cout)] : []
    }

    func callAsFunction(_ x: MLXArray, skips: [MLXArray], temb: MLXArray) -> MLXArray {
        var x = x
        for (i, r) in resnets.enumerated() {
            x = r(concatenated([x, skips[skips.count - 1 - i]], axis: -1), temb: temb)
        }
        if let u = upsamplers.first { x = u(x) }
        return x
    }
}

final class StyleRSIUpBlock2D: Module {
    let index: Int
    let sc_interpreter_offsets: [OffsetRefStrucInter]
    let dcn_deforms: [DeformConv2d]
    let resnets: [ResnetBlock2D]
    let attentions: [SpatialTransformer]
    let upsamplers: [Upsample2D]

    init(_ cin: Int, _ cprev: Int, _ cout: Int, temb: Int, eps: Float, addUpsample: Bool, contextDim: Int,
         index: Int, layers: Int = 3) {
        self.index = index
        var off = [OffsetRefStrucInter](), dcn = [DeformConv2d](), rs = [ResnetBlock2D](), at = [SpatialTransformer]()
        for i in 0 ..< layers {
            let skip = i == layers - 1 ? cin : cout
            off.append(OffsetRefStrucInter(resChannels: skip, styleChannels: 128 / index))
            dcn.append(DeformConv2d(skip))
            rs.append(ResnetBlock2D((i == 0 ? cprev : cout) + skip, cout, temb: temb, eps: eps))
            at.append(SpatialTransformer(channels: cout, heads: 1, dimHead: cout, contextDim: contextDim))
        }
        sc_interpreter_offsets = off
        dcn_deforms = dcn
        resnets = rs
        attentions = at
        upsamplers = addUpsample ? [Upsample2D(cout)] : []
    }

    func callAsFunction(_ x: MLXArray, skips: [MLXArray], temb: MLXArray, styleStructure: [MLXArray],
                        styleTokens: MLXArray) -> (MLXArray, MLXArray) {
        let ref = styleStructure[styleStructure.count - index - 2]
        var x = x
        var total = MLXArray(Float(0))
        for i in resnets.indices {
            let skip = skips[skips.count - 1 - i]
            let offset = sc_interpreter_offsets[i](skip, style: ref)
            total = total + abs(offset).mean()
            x = concatenated([x, dcn_deforms[i](skip, offset: offset)], axis: -1)
            x = attentions[i](resnets[i](x, temb: temb), context: styleTokens)
        }
        if let u = upsamplers.first { x = u(x) }
        return (x, total / Float(resnets.count))
    }
}

/// Encoder features for a batch: they do not depend on the diffusion step, so sampling computes them once.
public struct Conditions {
    /// (B, 9, 1024): the style encoder's 3x3 features, attended to by every style attention layer
    public var styleTokens: MLXArray
    /// content encoder on the content glyph: [input, 48 px, 24 px, 12 px, 12 px]; index 1...3 are used
    public var content: [MLXArray]
    /// content encoder on the style reference: same layout; index 1 and 2 feed the RSI blocks
    public var styleStructure: [MLXArray]

    /// All arrays in a fixed order (for compiled functions, which take flat lists).
    public var arrays: [MLXArray] { [styleTokens] + content + styleStructure }

    public init(styleTokens: MLXArray, content: [MLXArray], styleStructure: [MLXArray]) {
        self.styleTokens = styleTokens
        self.content = content
        self.styleStructure = styleStructure
    }

    /// Inverse of `arrays` (content and styleStructure always have the same length).
    public init(arrays: [MLXArray]) {
        let n = (arrays.count - 1) / 2
        styleTokens = arrays[0]
        content = Array(arrays[1 ..< 1 + n])
        styleStructure = Array(arrays[(1 + n)...])
    }
}

final class UNet: Module {
    let ds: Int
    let conv_in: Conv2d
    let time_embedding: TimestepEmbedding
    let down_blocks: [Module]
    let mid_block: UNetMidMCABlock2D
    let up_blocks: [Module]
    let conv_norm_out: GroupNorm
    let conv_out: Conv2d

    init(channels: [Int] = [64, 128, 256, 512], contentStart: Int = 64, contextDim: Int = 1024, eps: Float = 1e-5,
         ds: Int = 3) {
        let temb = channels[0] * 4
        let n = channels.count
        self.ds = ds
        conv_in = conv(3, channels[0], 3, padding: 1)
        time_embedding = TimestepEmbedding(channels[0], temb)
        var down = [Module]()
        var cout = channels[0]
        for i in 0 ..< n {
            let cin = cout
            cout = channels[i]
            let last = i == n - 1
            if i == 1 || i == 2 {
                down.append(MCADownBlock2D(cin, cout, temb: temb, eps: eps, addDownsample: !last,
                                           contentChannels: contentStart * (1 << (i - 1)), contextDim: contextDim))
            } else {
                down.append(DownBlock2D(cin, cout, temb: temb, eps: eps, addDownsample: !last))
            }
        }
        down_blocks = down
        mid_block = UNetMidMCABlock2D(channels[n - 1], temb: temb, eps: eps,
                                      contentChannels: contentStart * (1 << (ds - 1)), contextDim: contextDim)
        let rev = Array(channels.reversed())
        var up = [Module]()
        cout = rev[0]
        for i in 0 ..< n {
            let cprev = cout
            cout = rev[i]
            let cin = rev[min(i + 1, n - 1)]
            let last = i == n - 1
            if i == 1 || i == 2 {
                up.append(StyleRSIUpBlock2D(cin, cprev, cout, temb: temb, eps: eps, addUpsample: !last,
                                            contextDim: contextDim, index: i))
            } else {
                up.append(UpBlock2D(cin, cprev, cout, temb: temb, eps: eps, addUpsample: !last))
            }
        }
        up_blocks = up
        conv_norm_out = groupNorm(32, channels[0], eps)
        conv_out = conv(channels[0], 3, 3, padding: 1)
    }

    /// x: (B, H, W, 3) noisy glyphs; t: (B,) diffusion step (may be fractional).
    /// Returns (noise prediction, mean absolute deformable offset), as the PyTorch UNet does.
    func callAsFunction(_ x: MLXArray, t: MLXArray, cond: Conditions) -> (MLXArray, MLXArray) {
        let temb = time_embedding(timestepEmbedding(t, dim: conv_in.weight.dim(0)))
        var x = conv_in(x)
        var skips = [x]
        for (i, block) in down_blocks.enumerated() {
            let outs: [MLXArray]
            if let b = block as? MCADownBlock2D {
                (x, outs) = b(x, temb: temb, content: cond.content[i], styleTokens: cond.styleTokens)
            } else {
                (x, outs) = (block as! DownBlock2D)(x, temb: temb)
            }
            skips += outs
        }
        x = mid_block(x, temb: temb, content: cond.content[ds], styleTokens: cond.styleTokens)
        var offset = MLXArray(Float(0))
        for block in up_blocks {
            if let b = block as? StyleRSIUpBlock2D {
                let mine = Array(skips.suffix(b.resnets.count))
                skips.removeLast(b.resnets.count)
                let off: MLXArray
                (x, off) = b(x, skips: mine, temb: temb, styleStructure: cond.styleStructure,
                             styleTokens: cond.styleTokens)
                offset = offset + off
            } else {
                let b = block as! UpBlock2D
                let mine = Array(skips.suffix(b.resnets.count))
                skips.removeLast(b.resnets.count)
                x = b(x, skips: mine, temb: temb)
            }
        }
        return (conv_out(silu(conv_norm_out(x))), offset)
    }
}

// MARK: - encoders (BigGAN-style, spectral norm)

/// Conv with BigGAN spectral norm: weight / sigma, sigma from one power-iteration step on the stored `u0`.
///
/// Like the PyTorch original, sigma is recomputed on every call; in training mode the new `u0` (and `sv0`, kept
/// only so checkpoints round-trip) are stored back. Gradients flow through sigma but not through u and v.
final class SNConv2d: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray
    let u0: MLXArray
    let sv0: MLXArray
    let padding: Int
    let eps: Float

    init(_ cin: Int, _ cout: Int, k: Int = 3, padding: Int = 1, eps: Float = 1e-12) {
        let scale = 1 / sqrt(Float(cin * k * k))
        weight = MLXRandom.uniform(low: -scale, high: scale, [cout, k, k, cin])
        bias = MLXArray.zeros([cout])
        u0 = MLXRandom.normal([1, cout])
        sv0 = MLXArray.ones([1])
        self.padding = padding
        self.eps = eps
        super.init()
        freeze(recursive: false, keys: ["u0", "sv0"])
    }

    private func normalize(_ x: MLXArray) -> MLXArray {
        x / maximum(norm(x, axes: [1], keepDims: true), eps)
    }

    func effectiveWeight() -> MLXArray {
        let W = weight.reshaped(weight.dim(0), -1)       // flattening order does not change sigma
        let v = stopGradient(normalize(matmul(u0, W)))
        let u = stopGradient(normalize(matmul(v, W.T)))
        let sigma = matmul(matmul(v, W.T), u.T).reshaped([])
        if training {
            // in place, so compiled training steps (which track the model's arrays) see the new values
            u0._updateInternal(u)
            sv0._updateInternal(stopGradient(sigma).reshaped([1]))
        }
        return weight / sigma
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2d(x, effectiveWeight(), padding: .init(padding)) + bias
    }
}

final class DBlock: Module, UnaryLayer {
    let preactivation: Bool
    let conv1: SNConv2d
    let conv2: SNConv2d
    let conv_sc: SNConv2d

    init(_ cin: Int, _ cout: Int, preactivation: Bool) {
        self.preactivation = preactivation
        conv1 = SNConv2d(cin, cout)
        conv2 = SNConv2d(cout, cout)
        conv_sc = SNConv2d(cin, cout, k: 1, padding: 0)  // always learnable: every block downsamples
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h0 = preactivation ? relu(x) : x
        let h = avgPool2(conv2(relu(conv1(h0))))
        let sc = preactivation ? avgPool2(conv_sc(x)) : conv_sc(avgPool2(x))
        return h + sc
    }
}

/// 96 px: 3 blocks -> features at 48 (64 ch), 24 (128 ch), 12 (256 ch).
final class ContentEncoder: Module {
    let blocks: [[DBlock]]

    init(ch: Int = 64) {
        let cins = [3, ch, ch * 2], couts = [ch, ch * 2, ch * 4]
        blocks = (0 ..< 3).map { [DBlock(cins[$0], couts[$0], preactivation: $0 > 0)] }
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = x
        var feats = [x]
        for block in blocks {
            x = block[0](x)
            feats.append(x)
        }
        return (x, feats)
    }
}

/// 96 px: 5 blocks down to 3x3 (1024 ch), then (InstanceNorm, ReLU, 1x1 conv) applied twice, as in the original.
final class StyleEncoder: Module {
    let blocks: [[Module]]

    init(ch: Int = 64) {
        let cins = [3, ch, ch * 2, ch * 4, ch * 8], couts = [ch, ch * 2, ch * 4, ch * 8, ch * 16]
        var b: [[Module]] = (0 ..< 5).map { [DBlock(cins[$0], couts[$0], preactivation: $0 > 0)] }
        b.append([Op(), Op(), conv(couts[4], couts[4], 1)])
        blocks = b
    }

    private func last(_ x: MLXArray) -> MLXArray {
        (blocks[5][2] as! Conv2d)(relu(instanceNorm(x)))
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = x
        var feats = [x]
        for k in 0 ..< 5 {
            x = (blocks[k][0] as! DBlock)(x)
            if k < 4 { feats.append(x) }
        }
        return (last(last(x)), feats)
    }
}

// MARK: - the whole model

public final class FontDiffuser: Module {
    let unet: UNet
    let style_encoder: StyleEncoder
    let content_encoder: ContentEncoder

    public override init() {
        unet = UNet()
        style_encoder = StyleEncoder()
        content_encoder = ContentEncoder()
    }

    /// Encoder features for a batch of (content glyph, style reference) pairs, (B, 96, 96, 3) each in [-1, 1].
    public func encode(content: MLXArray, style: MLXArray) -> Conditions {
        let (sf, _) = style_encoder(style)
        let (cf, cFeats) = content_encoder(content)
        let (scf, scFeats) = content_encoder(style)
        return Conditions(styleTokens: sf.reshaped(sf.dim(0), -1, sf.dim(-1)),   // (B, 9, 1024)
                          content: cFeats + [cf], styleStructure: scFeats + [scf])
    }

    /// The denoiser only: (noise prediction, mean absolute deformable offset).
    public func denoise(_ x: MLXArray, t: MLXArray, cond: Conditions) -> (MLXArray, MLXArray) {
        unet(x, t: t, cond: cond)
    }

    public func callAsFunction(_ x: MLXArray, t: MLXArray, content: MLXArray, style: MLXArray) -> (MLXArray, MLXArray) {
        unet(x, t: t, cond: encode(content: content, style: style))
    }
}
