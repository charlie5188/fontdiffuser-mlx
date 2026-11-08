"""Time generation: N characters per batch, S steps, MLX with and without mx.compile for the denoiser call.
usage: python tests/bench_gen.py --weights W --font F --content-font C [--batch 8] [--steps 20,15]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdmlx.model import FontDiffuser
from fdmlx.weights import load_into
from fdmlx.sampler import Sampler
from fdmlx import glyphs as G
cli = argparse.ArgumentParser()
cli.add_argument('--weights', required=True); cli.add_argument('--font', required=True); cli.add_argument('--content-font', required=True)
cli.add_argument('--batch', type=int, default=8); cli.add_argument('--steps', default='20,15'); cli.add_argument('--repeat', type=int, default=3)
c = cli.parse_args()
chars = '我是道谢森风问家永国三酬袋鹰一十'[:c.batch]
model = FontDiffuser(); load_into(model, c.weights); model.eval()
content, style = G.glyphs(c.content_font, chars), G.glyphs(c.font, '国')
noise = mx.array(np.random.default_rng(0).standard_normal((len(chars), 96, 96, 3)).astype(np.float32))
for comp in (False, True):
    unet = mx.compile(lambda x, t, cond: model.unet(x, t, cond)[0]) if comp else None
    s = Sampler(model, unet=unet)
    for st in [int(x) for x in c.steps.split(',')]:
        s(content, style, noise, steps=st)                       # warm-up (and compile)
        ts = []
        for _ in range(c.repeat):
            t0 = time.time(); s(content, style, noise, steps=st); ts.append(time.time() - t0)
        print(f'batch {len(chars)}, {st} steps, compiled={comp}: {min(ts) / len(chars):.3f} s per character', flush=True)
    t0 = time.time()
    for _ in range(5): mx.eval((unet or (lambda x, t, cond: model.unet(x, t, cond)[0]))(mx.concatenate([noise, noise]), mx.full((2 * len(chars),), 500.), s.encode(content, mx.broadcast_to(style, content.shape))))
    print(f'   one denoiser call (batch {2 * len(chars)}): {(time.time() - t0) / 5:.3f} s', flush=True)
