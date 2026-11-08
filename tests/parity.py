"""Check the MLX port against the official PyTorch code: same weights, same inputs, compare outputs and gradients.

Needs PyTorch, torchvision, diffusers 0.22 and a clone of the official repo (with its ckpt/ folder):
  FONTDIFFUSER_REPO=/path/to/FontDiffuser python tests/parity.py [--finetuned last.pth]
PyTorch runs on the CPU in fp32 as the reference; MLX runs on the GPU in fp32.
"""
import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdmlx.deform import deform_conv2d  # noqa: E402
from fdmlx.model import FontDiffuser  # noqa: E402
from fdmlx.weights import load_into, load_torch_checkpoint, torch_to_mlx  # noqa: E402

REPO = Path(os.environ["FONTDIFFUSER_REPO"])
sys.path.insert(0, str(REPO))
from configs.fontdiffuser import get_parser  # noqa: E402
from src import build_content_encoder, build_style_encoder, build_unet  # noqa: E402
from src.model import FontDiffuserModel  # noqa: E402

cli = argparse.ArgumentParser()
cli.add_argument("--finetuned", default=None)
cli.add_argument("--batch", type=int, default=2)
c = cli.parse_args()

fails = []


def check(name, got, want, tol=1e-4):
    got = np.array(got, dtype=np.float64)
    want = want.detach().double().numpy() if torch.is_tensor(want) else np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        got = got.transpose(0, 3, 1, 2) if got.ndim == 4 else got        # MLX channels-last -> PyTorch layout
    err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
    ok = err < tol
    print(f"{'ok  ' if ok else 'FAIL'} {name:42s} rel err {err:.2e}  (shape {tuple(want.shape)})")
    if not ok:
        fails.append(name)


def nhwc(t):
    return mx.array(t.detach().numpy().transpose(0, 2, 3, 1))


# ---------------------------------------------------------------- 1. deformable conv, forward and backward
from torchvision.ops import deform_conv2d as tv_deform  # noqa: E402

rng = np.random.default_rng(0)
x = rng.standard_normal((2, 8, 12, 12)).astype(np.float32)
off = (rng.standard_normal((2, 18, 12, 12)) * 3).astype(np.float32)      # large offsets: many taps fall outside
w = rng.standard_normal((8, 8, 3, 3)).astype(np.float32)
b = rng.standard_normal(8).astype(np.float32)
tx, toff, tw = (torch.tensor(a, requires_grad=True) for a in (x, off, w))
ty = tv_deform(tx, toff, tw, torch.tensor(b), padding=1)
g = rng.standard_normal(ty.shape).astype(np.float32)
(ty * torch.tensor(g)).sum().backward()


def f(xx, oo, ww):
    return (deform_conv2d(xx, oo, ww, mx.array(b), padding=1) * mx.array(g.transpose(0, 2, 3, 1))).sum()


my = deform_conv2d(mx.array(x.transpose(0, 2, 3, 1)), mx.array(off.transpose(0, 2, 3, 1)),
                   mx.array(torch_to_mlx(w)), mx.array(b), padding=1)
check("deform conv forward", my, ty)
gx, goff, gw = mx.grad(f, argnums=(0, 1, 2))(mx.array(x.transpose(0, 2, 3, 1)), mx.array(off.transpose(0, 2, 3, 1)),
                                              mx.array(torch_to_mlx(w)))
check("deform conv grad input", gx, tx.grad)
check("deform conv grad offset", goff, toff.grad, tol=1e-3)
check("deform conv grad weight", np.array(gw).transpose(0, 3, 1, 2), tw.grad)

# ---------------------------------------------------------------- 2. the whole model with real weights
args = get_parser().parse_args([])
args.style_image_size = (96, 96)
args.content_image_size = (96, 96)


def load(build, name):
    m = build(args=args)
    m.load_state_dict(torch.load(REPO / "ckpt" / name, map_location="cpu", weights_only=True))
    return m


tm = FontDiffuserModel(unet=load(build_unet, "unet.pth"), style_encoder=load(build_style_encoder, "style_encoder.pth"),
                       content_encoder=load(build_content_encoder, "content_encoder.pth"))
if c.finetuned:
    sd = torch.load(c.finetuned, map_location="cpu", weights_only=True)
    for k in ("unet", "style_encoder", "content_encoder"):
        getattr(tm, k).load_state_dict(sd[k])
tm.eval()

mm = FontDiffuser()
load_into(mm, load_torch_checkpoint(ckpt_dir=REPO / "ckpt", finetuned=c.finetuned))
mm.eval()

B = c.batch
torch.manual_seed(0)
# glyph-like inputs: mostly white with dark strokes, in [-1, 1]
content = torch.where(torch.rand(B, 1, 96, 96) < 0.15, -1.0, 1.0).expand(-1, 3, -1, -1).contiguous()
style = torch.where(torch.rand(B, 1, 96, 96) < 0.2, -1.0, 1.0).expand(-1, 3, -1, -1).contiguous()
xt = torch.randn(B, 3, 96, 96)
ts = torch.tensor([999.0, 321.5][:B] + [10.0] * max(0, B - 2))

with torch.no_grad():
    sf, _, _ = tm.style_encoder(style)
    cf, cres = tm.content_encoder(content)
    want_eps, want_off = tm(xt, ts, style, content, 3)
msf, _ = mm.style_encoder(nhwc(style))
mcf, mres = mm.content_encoder(nhwc(content))
check("style encoder output", msf, sf)
check("content encoder output", mcf, cf)
for i, (a, bb) in enumerate(zip(mres, cres)):
    check(f"content encoder feature {i}", a, bb)
eps, offv = mm(nhwc(xt), mx.array(ts.numpy()), nhwc(content), nhwc(style))
check("UNet noise prediction", eps, want_eps)
check("UNet offset sum", np.array(offv).reshape(()), want_off.reshape(()))

# ---------------------------------------------------------------- 3. one training step: loss and gradients
tm.train()
mm.train()
noise = torch.randn(B, 3, 96, 96)
pred, o = tm(xt, ts, style, content, 3)
tloss = torch.nn.functional.mse_loss(pred, noise) + 0.5 * (o / 2)
tm.zero_grad()
tloss.backward()


def loss_fn(model):
    p, oo = model(nhwc(xt), mx.array(ts.numpy()), nhwc(content), nhwc(style))
    return ((p - nhwc(noise)) ** 2).mean() + 0.5 * (oo / 2)


mloss, grads = nn.value_and_grad(mm, loss_fn)(mm)
check("training loss", np.array(mloss).reshape(()), tloss.detach().reshape(()), tol=1e-4)
from mlx.utils import tree_flatten  # noqa: E402

mg = dict(tree_flatten(grads))
tg = {f"{part}.{k}": p.grad for part in ("unet", "style_encoder", "content_encoder")
      for k, p in getattr(tm, part).named_parameters() if p.grad is not None}
missing = sorted(set(tg) - set(mg))
print(f"grads: {len(mg)} MLX, {len(tg)} PyTorch, missing in MLX: {missing[:5]}")
# Judge every gradient against the typical gradient size: some true gradients are ~0 (a bias right before
# InstanceNorm, cross-attention queries when the softmax over the 9 style tokens is saturated), and there both
# sides are rounding noise. The offset branch of the RSI blocks is noisy for another reason: the released model's
# offsets are ~1e-4 px, so its gradients are sums of large, nearly cancelling terms. Against a float64 run of the
# official code, official fp32 is itself ~5% off there (MLX fp32: the same), so the worst-case bound is 0.1.
scale = np.median([np.abs(t.numpy()).max() for t in tg.values()])
errs = {k: float(np.abs(np.array(mg[k]) - torch_to_mlx(tg[k].numpy())).max()) / max(np.abs(tg[k].numpy()).max(), 1e-3 * scale)
        for k in tg}
worst = max(errs, key=errs.get)
med = float(np.median(list(errs.values())))
ok = med < 1e-4 and errs[worst] < 0.1
print(f"{'ok  ' if ok else 'FAIL'} {'all ' + str(len(tg)) + ' parameter gradients':42s} median rel err {med:.1e}, "
      f"worst {errs[worst]:.1e} ({worst})")
if not ok:
    fails.append("gradients")
# spectral norm buffers after one training forward: u0 must match PyTorch's updated u0
tu = tm.style_encoder.blocks[1][0].conv1.u0
check("SN u0 after a training step", mm.style_encoder.blocks[1][0].conv1.u0, tu, tol=1e-4)

print("\nALL OK" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
