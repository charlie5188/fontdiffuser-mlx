// Deformable convolution v1 (torchvision.ops.deform_conv2d, one offset group, no mask), from gathers and a
// matmul. A translation of fdmlx/deform.py; see there for the out-of-range rules, which match torchvision.

import MLX

/// x: (B, H, W, C); offset: (B, Ho, Wo, 2*kh*kw), (dy, dx) per tap in row-major tap order (torchvision's layout);
/// weight: (Co, kh, kw, C); bias: (Co,). Returns (B, Ho, Wo, Co).
func deformConv2d(_ x: MLXArray, offset: MLXArray, weight: MLXArray, bias: MLXArray?, padding: Int = 1,
                  stride: Int = 1, dilation: Int = 1) -> MLXArray {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let (Co, kh, kw) = (weight.dim(0), weight.dim(1), weight.dim(2))
    let K = kh * kw
    let (Ho, Wo) = (offset.dim(1), offset.dim(2))

    let tapY = MLXArray((0 ..< K).map { Float(($0 / kw) * dilation) })
    let tapX = MLXArray((0 ..< K).map { Float(($0 % kw) * dilation) })
    let baseY = MLXArray((0 ..< Ho).map { Float($0 * stride - padding) }).reshaped(Ho, 1, 1) + tapY   // (Ho, 1, K)
    let baseX = MLXArray((0 ..< Wo).map { Float($0 * stride - padding) }).reshaped(1, Wo, 1) + tapX   // (1, Wo, K)
    let off = offset.asType(.float32).reshaped(B, Ho, Wo, K, 2)
    let py = baseY + off[.ellipsis, 0]                                                                 // (B, Ho, Wo, K)
    let px = baseX + off[.ellipsis, 1]

    let y0f = floor(py), x0f = floor(px)
    let fy = py - y0f, fx = px - x0f
    let y0 = stopGradient(y0f).asType(.int32), x0 = stopGradient(x0f).asType(.int32)  // corners carry no gradient

    let flat = x.reshaped(B, H * W, C)
    let batch = MLXArray(0 ..< B).reshaped(B, 1)
    let corners: [(Int, Int, MLXArray)] = [(0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                                          (1, 0, fy * (1 - fx)), (1, 1, fy * fx)]
    var cols: MLXArray?
    for (dy, dx, w) in corners {
        let yy = y0 + dy, xx = x0 + dx
        let inside = (yy .>= 0) .&& (yy .< H) .&& (xx .>= 0) .&& (xx .< W)
        let idx = (clip(yy, min: 0, max: H - 1) * W + clip(xx, min: 0, max: W - 1)).reshaped(B, -1)  // (B, Ho*Wo*K)
        let wm = which(inside, w, Float(0)).reshaped(B, -1, 1).asType(x.dtype)
        let sampled = flat[batch, idx] * wm                                                          // (B, Ho*Wo*K, C)
        cols = cols.map { $0 + sampled } ?? sampled
    }

    // tap-major, channel-minor = the weight's (kh, kw, C) order
    var out = matmul(cols!.reshaped(B, Ho * Wo, K * C), weight.reshaped(Co, K * C).T)
    if let bias { out = out + bias }
    return out.reshaped(B, Ho, Wo, Co)
}
