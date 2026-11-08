"""Make a Core ML template of FontDiffuser's denoiser (the UNet) plus a map of where each weight is stored in it.

Run once, at development time. Afterwards any weights (official, or fine-tuned in MLX) go into a copy of the
template with `fdmlx coreml-swap`, which only overwrites bytes and needs neither PyTorch nor coremltools.

Why a template with placeholder weights: Core ML's converter only reads PyTorch (or TensorFlow) models. Converting
different weights never changes the Core ML program, only the numbers in its weight file (checked: a swapped file
is byte-identical to a fresh conversion). The template is converted from random weights that are all different
from each other, so every stored weight can be matched to exactly one parameter by its bytes.

Needs: torch <= 2.7, coremltools 9, diffusers 0.22, and a clone of the official FontDiffuser repository (only its
model code is used, not its weights):
  python scripts/coreml_export.py --repo /path/to/FontDiffuser --out unet-b8.mlpackage [--chars 8]
Writes unet-b8.mlpackage (placeholder weights, not usable on its own) and unet-b8.weights-map.json.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cli = argparse.ArgumentParser()
cli.add_argument("--repo", required=True, help="clone of the official FontDiffuser repository")
cli.add_argument("--out", required=True, help="output .mlpackage")
cli.add_argument("--chars", type=int, default=8, help="characters per call (fixed; the batch is twice this with guidance)")
c = cli.parse_args()
sys.path.insert(0, c.repo)

import coremltools as ct  # noqa: E402
from coremltools.proto import MIL_pb2  # noqa: E402
from configs.fontdiffuser import get_parser  # noqa: E402
from src import build_unet  # noqa: E402
from torchvision.ops import DeformConv2d  # noqa: E402


def deform_conv2d(x, offset, weight, bias, padding=1):
    """torchvision's deform_conv2d (one offset group, no mask) from grid_sample + matmul: Core ML has grid_sample
    but no deformable convolution."""
    B, C, H, W = x.shape
    Co, _, kh, kw = weight.shape
    K = kh * kw
    Ho, Wo = offset.shape[-2:]
    ky, kx = torch.meshgrid(torch.arange(kh), torch.arange(kw), indexing="ij")
    ky, kx = ky.reshape(K, 1, 1).float(), kx.reshape(K, 1, 1).float()
    oy = torch.arange(Ho).float().view(1, Ho, 1) - padding
    ox = torch.arange(Wo).float().view(1, 1, Wo) - padding
    off = offset.view(B, K, 2, Ho, Wo)
    py, px = oy + ky + off[:, :, 0], ox + kx + off[:, :, 1]
    grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], dim=-1).view(B, K * Ho, Wo, 2)
    cols = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(B, C * K, Ho * Wo)
    return (weight.view(Co, C * K) @ cols).view(B, Co, Ho, Wo) + bias.view(1, Co, 1, 1)


args = get_parser().parse_args([])
unet = build_unet(args).eval()
for m in unet.modules():
    if isinstance(m, DeformConv2d):
        m.forward = (lambda mod: lambda inp, off, mask=None: deform_conv2d(inp, off, mod.weight, mod.bias))(m)

# placeholder weights: random, every parameter different, so the stored bytes identify the parameter
g = torch.Generator().manual_seed(0)
params = dict(unet.named_parameters())
with torch.no_grad():
    for p in params.values():
        p.copy_(torch.randn(p.shape, generator=g) * 0.05)


class UNetFlat(nn.Module):
    """The UNet with only the encoder features it uses as plain inputs (Core ML needs a flat signature).
    style_tokens (B, 9, 1024): style encoder output; content1..3: content encoder features at 48, 24, 12 px;
    structure0..1: content encoder features of the style reference at 48 and 24 px."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, style_tokens, content1, content2, content3, structure0, structure1):
        B = x.shape[0]
        sf = style_tokens.reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        hs = [sf, [content1, content1, content2, content3], style_tokens, [structure0, structure1, structure1, structure1]]
        return self.unet(x, t, encoder_hidden_states=hs, content_encoder_downsample_size=3)[0]


B = 2 * c.chars
shapes = {"x": (B, 3, 96, 96), "t": (B,), "style_tokens": (B, 9, 1024), "content1": (B, 64, 48, 48),
          "content2": (B, 128, 24, 24), "content3": (B, 256, 12, 12), "structure0": (B, 64, 48, 48),
          "structure1": (B, 128, 24, 24)}
ex = [torch.randn(s) if k != "t" else torch.full(s, 500.0) for k, s in shapes.items()]
net = UNetFlat(unet).eval()
with torch.no_grad():
    traced = torch.jit.trace(net, tuple(ex), check_trace=False)

# coremltools 9.0 casts one-element arrays with int(array), which numpy 2 refuses; cast via .item() instead
from coremltools.converters.mil.frontend.torch import ops as _ops  # noqa: E402
_orig_cast = _ops._cast


def _cast(context, node, dtype, dtype_name):
    x = _ops._get_inputs(context, node, expected=1)[0]
    if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
        return context.add(_ops.mb.const(val=dtype(x.val.item()), name=node.name), node.name)
    return _orig_cast(context, node, dtype, dtype_name)


_ops._cast = _cast

pkg = Path(c.out)
ml = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                inputs=[ct.TensorType(name=k, shape=s) for k, s in shapes.items()],
                outputs=[ct.TensorType(name="noise")], compute_precision=ct.precision.FLOAT16)
ml.save(str(pkg))

# ---------------------------------------------------------------- the weight map
data = pkg / "Data/com.apple.CoreML"
spec = ct.utils.load_spec(str(pkg))
fn = spec.mlProgram.functions["main"]
ops = fn.block_specializations[fn.opset].operations
lookup = {}
for name, p in params.items():
    a = p.detach().numpy()
    lookup[a.astype(np.float16).tobytes()] = (name, "same")
    if a.ndim == 2:
        lookup[a.T.astype(np.float16).tobytes()] = (name, "T")

entries, seen = [], set()
wb = (data / "weights/weight.bin").read_bytes()
for op in ops:
    v = op.attributes["val"] if op.type == "const" else None
    if v is None or not v.HasField("blobFileValue"):
        continue
    sentinel, dtype, size, off = struct.unpack("<IIQQ", wb[v.blobFileValue.offset:v.blobFileValue.offset + 24])
    assert sentinel == 0xDEADBEEF and dtype == 1, "expected fp16 weight blobs"
    hit = lookup.get(wb[off:off + size])
    if hit:
        entries.append({"param": f"unet.{hit[0]}", "file": "weights/weight.bin", "offset": off, "nbytes": size,
                        "layout": hit[1]})
        seen.add(hit[0])

mlmodel = (data / "model.mlmodel").read_bytes()
for op in ops:               # a few tiny parameters are stored inside the program file instead of weight.bin
    v = op.attributes["val"] if op.type == "const" else None
    if v is None or not v.HasField("immediateValue") or v.type.tensorType.dataType != MIL_pb2.FLOAT16:
        continue
    raw = bytes(v.immediateValue.tensor.bytes.values)
    hit = lookup.get(raw)
    if hit and hit[0] not in seen:
        pos = mlmodel.find(raw)
        assert pos >= 0 and mlmodel.find(raw, pos + 1) < 0, f"{hit[0]}: bytes not unique in model.mlmodel"
        entries.append({"param": f"unet.{hit[0]}", "file": "model.mlmodel", "offset": pos, "nbytes": len(raw),
                        "layout": hit[1]})
        seen.add(hit[0])

missing = sorted(set(params) - seen)
assert not missing, f"parameters with no place in the Core ML file: {missing}"
info = {"chars_per_call": c.chars, "inputs": {k: list(s) for k, s in shapes.items()}, "output": "noise",
        "layout": "channels-first (N, C, H, W), float32 in, fp16 inside",
        "template_sha256": {f: hashlib.sha256((data / f).read_bytes()).hexdigest()
                            for f in ("model.mlmodel", "weights/weight.bin")},
        "weights": entries}
Path(str(pkg).removesuffix(".mlpackage") + ".weights-map.json").write_text(json.dumps(info, indent=1))
print(f"{pkg}: {len(entries)} parameters mapped ({sum(e['file'] == 'model.mlmodel' for e in entries)} in the program file)")
