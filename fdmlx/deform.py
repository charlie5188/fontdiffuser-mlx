"""Deformable convolution v1 (torchvision.ops.deform_conv2d with one offset group and no mask) for MLX.

MLX has no deformable-conv op. This builds it from gathers and one matmul, channels-last:
for every output position and kernel tap, sample the input at (position + tap + learned offset) with bilinear
interpolation, then multiply the sampled columns by the weight. Out-of-range rules match torchvision exactly:
each of the 4 bilinear corners contributes only if it lies inside the image, so a sample point at or beyond one
pixel outside the border reads 0. Gradients reach the input (through the gather) and the offsets (through the
bilinear weights), like torchvision's backward pass.
"""
import mlx.core as mx


def deform_conv2d(x, offset, weight, bias=None, padding=1, stride=1, dilation=1):
    """x: (B, H, W, C); offset: (B, Ho, Wo, 2*kh*kw), (dy, dx) per tap in row-major tap order (torchvision's layout);
    weight: (Co, kh, kw, C); bias: (Co,). Returns (B, Ho, Wo, Co)."""
    B, H, W, C = x.shape
    Co, kh, kw, _ = weight.shape
    K = kh * kw
    _, Ho, Wo, _ = offset.shape

    tap_y = (mx.arange(K) // kw).astype(mx.float32) * dilation
    tap_x = (mx.arange(K) % kw).astype(mx.float32) * dilation
    base_y = (mx.arange(Ho) * stride - padding).astype(mx.float32)[:, None, None] + tap_y      # (Ho, 1, K)
    base_x = (mx.arange(Wo) * stride - padding).astype(mx.float32)[None, :, None] + tap_x      # (1, Wo, K)
    off = offset.astype(mx.float32).reshape(B, Ho, Wo, K, 2)
    py = base_y + off[..., 0]                                                                  # (B, Ho, Wo, K)
    px = base_x + off[..., 1]

    y0, x0 = mx.floor(py), mx.floor(px)
    fy, fx = py - y0, px - x0
    y0, x0 = mx.stop_gradient(y0).astype(mx.int32), mx.stop_gradient(x0).astype(mx.int32)   # integer corners carry no gradient

    flat = x.reshape(B, H * W, C)
    batch = mx.arange(B)[:, None]
    cols = 0
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        idx = (mx.clip(yy, 0, H - 1) * W + mx.clip(xx, 0, W - 1)).reshape(B, -1)              # (B, Ho*Wo*K)
        w = mx.where(inside, w, 0.0).reshape(B, -1, 1).astype(x.dtype)
        cols = cols + flat[batch, idx] * w                                                     # (B, Ho*Wo*K, C)

    cols = cols.reshape(B, Ho * Wo, K * C)                       # tap-major, channel-minor = weight's (kh, kw, C)
    out = cols @ weight.reshape(Co, K * C).T
    if bias is not None:
        out = out + bias
    return out.reshape(B, Ho, Wo, Co)
