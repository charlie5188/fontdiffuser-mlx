"""Fine-tune FontDiffuser on one font's glyphs, in MLX.

Recipe (the official phase-1 objective, at fine-tuning scale):
  sample = (target glyph X in the target font, X in the plain content font, another training glyph Y in the
  target font as the style reference); loss = noise MSE + 0.5 * mean |deformable offset| / 2;
  10 % of samples get blank content and style images, so classifier-free guidance keeps working;
  AdamW (lr 2e-5, weight decay 0.01), gradient norm clipped to 1.
Left out compared with the paper's training: the VGG perceptual loss (weight 0.01) and the phase-2 SCR module.

Every --eval-every steps it saves the weights and draws the held-out test characters with each eval reference,
so you can see whether the model improves, stalls or starts to over-fit. Test characters never enter training.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from . import glyphs as G
from .model import FontDiffuser
from .sampler import Sampler, alphas_cumprod
from .weights import load_into, save
from mlx.utils import tree_map


def build_parser():
    p = argparse.ArgumentParser(prog="fdmlx finetune", description=__doc__.split("\n")[0])
    p.add_argument("--weights", required=True, help="start from: official ckpt folder, a .pth checkpoint or .safetensors")
    p.add_argument("--font", required=True, help="the target font (.ttf / .otf)")
    p.add_argument("--train", required=True, help="text file with the training characters")
    p.add_argument("--content-font", required=True, help="plain font for the character skeletons (the official model "
                   "was trained with KaiXinSongA.ttf from the FontDiffuser repo)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--test", default="", help="held-out characters to draw at every evaluation (never trained on)")
    p.add_argument("--eval-refs", default="", help="style references for evaluation (must be training characters)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=4, help="split each batch into this many passes: same update, less memory "
                   "(batch 16 needs about 19 GB in one pass, 10 GB in 2, 7 GB in 4, 4 GB in 8; speed per glyph is the same)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-steps", type=int, default=20, help="sampling steps for the evaluation sheets")
    p.add_argument("--eval-noise", default=None, help="optional .npy (1, 96, 96, 3) starting noise for the evaluation sheets, "
                   "e.g. to compare with another run; default: a fixed noise from seed 123")
    p.add_argument("--no-compile", action="store_true", help="run the training step without mx.compile")
    return p


def main(argv=None):
    c = build_parser().parse_args(argv)
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=True)
    test = c.test
    train = [ch for ch in dict.fromkeys(Path(c.train).read_text(encoding="utf-8")) if not ch.isspace() and ch not in test]
    for f in (c.font, c.content_font):
        missing = [ch for ch in train + list(test) if not G.has_glyph(f, ch)]
        if missing:
            raise SystemExit(f"{f} has no glyph for: {''.join(missing)}")
    assert c.batch % c.accum == 0, "--batch must be divisible by --accum"
    refs = c.eval_refs or "".join(train[:1])
    assert all(r in train for r in refs), "eval refs must be training characters"
    print(f"{len(train)} training characters, {len(test)} held-out test characters", flush=True)

    TGT = G.glyphs(c.font, "".join(train))
    CON = G.glyphs(c.content_font, "".join(train))
    TEST_CON = G.glyphs(c.content_font, test) if test else None
    REF = {r: G.glyphs(c.font, r) for r in refs}

    model = FontDiffuser()
    load_into(model, c.weights)
    model.train()
    opt = optim.AdamW(learning_rate=c.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=1e-2, bias_correction=True)
    acp = mx.array(alphas_cumprod())
    rng = np.random.default_rng(c.seed)

    def loss_fn(model, noisy, t, con, sty, noise):
        pred, off = model(noisy, t, con, sty)
        return ((pred - noise) ** 2).mean() + 0.5 * (off / 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def micro_fn(tgt, con, sty, noise, t):
        a = acp[t][:, None, None, None]
        noisy = mx.sqrt(a) * tgt + mx.sqrt(1 - a) * noise
        return grad_fn(model, noisy, t, con, sty, noise)

    def update_fn(grads):
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)

    # Each pass is evaluated before the next one starts, so its activations are freed: peak memory follows the
    # pass size, not the batch size. (One compiled graph for all passes would keep them all alive at once.)
    # model.state is an output of the pass too: the spectral-norm vectors update in training mode.
    micro = micro_fn if c.no_compile else mx.compile(micro_fn, inputs=model.state, outputs=model.state)
    update = update_fn if c.no_compile else mx.compile(update_fn, inputs=[model.state, opt.state],
                                                        outputs=[model.state, opt.state])
    state = [model.state, opt.state]

    def step(tgt, con, sty, noise, t):
        parts = c.batch // c.accum
        loss, grads = 0.0, None
        for k in range(0, c.batch, parts):
            sl = slice(k, k + parts)
            l, g = micro(tgt[sl], con[sl], sty[sl], noise[sl], t[sl])
            loss = loss + l / c.accum
            grads = g if grads is None else tree_map(mx.add, grads, g)
            mx.eval(loss, grads, model.state)
        if c.accum > 1:
            grads = tree_map(lambda x: x / c.accum, grads)
        update(grads)
        return loss

    def evaluate(n):
        if not test:
            return
        model.eval()
        s = Sampler(model)
        noise = np.load(c.eval_noise) if c.eval_noise else np.random.default_rng(123).standard_normal((1, 96, 96, 3))
        noise = mx.array(noise.astype(np.float32))
        noise = mx.broadcast_to(noise, TEST_CON.shape)        # same start for every character and every evaluation
        for r, ref in REF.items():
            G.sheet(G.to_images(s(TEST_CON, ref, noise, steps=c.eval_steps))).save(out / f"grid-step{n:04d}-ref{r}.png")
        model.train()

    log = []
    evaluate(0)
    t0, run = time.time(), None
    n_train = len(train)
    for i in range(1, c.steps + 1):
        idx = rng.integers(0, n_train, c.batch)
        sidx = np.array([rng.choice([j for j in range(n_train) if j != k]) for k in idx])
        drop = rng.random(c.batch) < 0.1                      # condition dropout keeps classifier-free guidance working
        con, sty = CON[mx.array(idx)], TGT[mx.array(sidx)]
        d = mx.array(drop)[:, None, None, None]
        con, sty = mx.where(d, 1.0, con), mx.where(d, 1.0, sty)
        noise = mx.array(rng.standard_normal((c.batch, 96, 96, 3)).astype(np.float32))
        t = mx.array(rng.integers(0, 1000, c.batch).astype(np.int32))
        loss = step(TGT[mx.array(idx)], con, sty, noise, t)
        mx.eval(state, loss)
        lv = loss.item()
        run = lv if run is None else 0.95 * run + 0.05 * lv
        if i % 20 == 0 or i == c.steps:
            el = time.time() - t0
            print(f"step {i}/{c.steps} loss {run:.4f}  {el / i:.2f}s/step  eta {(c.steps - i) * el / i / 60:.0f} min  "
                  f"peak mem {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
            log.append({"step": i, "loss": round(run, 5), "elapsed_s": round(el, 1)})
        if i % c.eval_every == 0 or i == c.steps:
            save(model, out / "weights.safetensors")
            evaluate(i)
            json.dump({"train_chars": "".join(train), "args": vars(c), "log": log},
                      open(out / "log.json", "w"), ensure_ascii=False, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out / 'weights.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
