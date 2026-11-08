"""Reference outputs from the Python version, for checking the Swift version (swift/checks, `fdmlx-swift-check`).

  python tests/swift_fixtures.py --weights W.safetensors --content-font KaiXinSongA.ttf --font F.ttf \
      --image glyph.png --out fixtures.safetensors

Writes the inputs and the Python results for: glyph rendering, the photo-to-reference step, both encoders, one
denoiser call, a whole sampling run, the training loss and every parameter gradient, and one full training step
(2 passes, gradient clipping, AdamW, spectral-norm updates). The file is large (about 800 MB: two copies of the
weights' size), so write it somewhere temporary. Run from the repository root (PYTHONPATH=. if fdmlx is not
installed).
"""
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from fdmlx import glyphs as G
from fdmlx.cli import ref_from_image
from fdmlx.model import FontDiffuser
from fdmlx.sampler import Sampler, alphas_cumprod
from fdmlx.weights import load_into

CHARS, REF = "我是道谢", "永"


def main():
    p = argparse.ArgumentParser()
    for k in ("--weights", "--content-font", "--font", "--image", "--out"):
        p.add_argument(k, required=True)
    c = p.parse_args()
    rng = np.random.default_rng(0)
    out = {}

    # rendering: 128 px renders (to check the 128 -> 96 resize alone) and the model inputs
    out["render128"] = mx.array(np.stack([np.asarray(G.render(f, ch)) for f, ch in
                                          [(c.content_font, ch) for ch in CHARS] + [(c.font, REF)]]))
    content, style = G.glyphs(c.content_font, CHARS), G.glyphs(c.font, REF)
    out["content"], out["style"] = content, style
    out["refimg"] = ref_from_image(c.image)

    model = FontDiffuser()
    load_into(model, c.weights)
    model.eval()
    styles = mx.broadcast_to(style, content.shape)
    cond = model.encode(content, styles)
    out["enc.style_tokens"] = cond["style_tokens"]
    for i, a in enumerate(cond["content"]):
        out[f"enc.content.{i}"] = a
    for i, a in enumerate(cond["style_structure"]):
        out[f"enc.style_structure.{i}"] = a
    x = mx.array(rng.standard_normal((4, 96, 96, 3)).astype(np.float32))
    t = mx.array([999.0, 500.0, 10.5, 0.0])
    eps, off = model.unet(x, t, cond)
    out.update({"unet.x": x, "unet.t": t, "unet.eps": eps, "unet.offset": off})

    noise = mx.array(rng.standard_normal((4, 96, 96, 3)).astype(np.float32))
    out["sample.noise"] = noise
    out["sample"] = mx.array(Sampler(model)(content, style, noise, steps=15))

    # training: loss and gradients in training mode, from freshly loaded weights
    tgt = G.glyphs(c.font, "国东力今")
    sty = G.glyphs(c.font, "三十一量")
    tn = mx.array(rng.standard_normal((4, 96, 96, 3)).astype(np.float32))
    tt = mx.array(np.array([5, 250, 600, 990], dtype=np.int32))
    out.update({"train.tgt": tgt, "train.con": content, "train.sty": sty, "train.noise": tn, "train.t": tt})
    acp = mx.array(alphas_cumprod())

    def loss_fn(model, noisy, t, con, sty, noise):
        pred, off = model(noisy, t, con, sty)
        return ((pred - noise) ** 2).mean() + 0.5 * (off / 2)

    def noisy_of(tgt, noise, t):
        a = acp[t][:, None, None, None]
        return mx.sqrt(a) * tgt + mx.sqrt(1 - a) * noise

    model = FontDiffuser()
    load_into(model, c.weights)
    model.train()
    loss, grads = nn.value_and_grad(model, loss_fn)(model, noisy_of(tgt, tn, tt), tt, content, sty, tn)
    out["train.loss"] = loss
    for k, g in tree_flatten(grads):
        out[f"grad.{k}"] = g

    # one training step exactly as fdmlx/finetune.py does it: batch 4 in 2 passes, compiled
    model = FontDiffuser()
    load_into(model, c.weights)
    model.train()
    opt = optim.AdamW(learning_rate=2e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=1e-2, bias_correction=True)
    grad_fn = nn.value_and_grad(model, loss_fn)

    def micro_fn(tgt, con, sty, noise, t):
        return grad_fn(model, noisy_of(tgt, noise, t), t, con, sty, noise)

    def update_fn(grads):
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)

    micro = mx.compile(micro_fn, inputs=model.state, outputs=model.state)
    update = mx.compile(update_fn, inputs=[model.state, opt.state], outputs=[model.state, opt.state])
    total, acc = 0.0, None
    for k in (0, 2):
        sl = slice(k, k + 2)
        l, g = micro(tgt[sl], content[sl], sty[sl], tn[sl], tt[sl])
        total = total + l / 2
        acc = g if acc is None else tree_map(mx.add, acc, g)
        mx.eval(total, acc, model.state)
    update(tree_map(lambda a: a / 2, acc))
    mx.eval(model.state, opt.state)
    out["step.loss"] = total
    for k, v in tree_flatten(model.parameters()):
        out[f"step.after.{k}"] = v

    mx.save_safetensors(c.out, out)
    print(f"wrote {len(out)} arrays to {c.out}")


if __name__ == "__main__":
    main()
