"""Core ML vs MLX: one denoiser call on the same inputs, then the 8 test characters end to end, and timing.
usage: python tests/coreml_check.py --package P --weights W --font F --content-font C --noise N --out DIR"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdmlx.model import FontDiffuser
from fdmlx.weights import load_into
from fdmlx.sampler import Sampler
from fdmlx.coreml import CoreMLUNet
from fdmlx import glyphs as G
cli = argparse.ArgumentParser()
for a in ('--package', '--weights', '--font', '--content-font', '--noise', '--out'): cli.add_argument(a, required=True)
cli.add_argument('--refs', default='国永'); cli.add_argument('--units', default='CPU_AND_GPU,CPU_AND_NE')
c = cli.parse_args(); out = Path(c.out); out.mkdir(parents=True, exist_ok=True)
chars = '我是道谢森风问家'
model = FontDiffuser(); load_into(model, c.weights); model.eval()
content = G.glyphs(c.content_font, chars)
noise = mx.broadcast_to(mx.array(np.load(c.noise).astype(np.float32)), content.shape)
ref = {r: G.glyphs(c.font, r) for r in c.refs}
mlx_s = Sampler(model)
cond = mlx_s.encode(content, mx.broadcast_to(ref[c.refs[0]], content.shape))
x2 = mx.array(np.random.default_rng(0).standard_normal((16, 96, 96, 3)).astype(np.float32)); t2 = mx.full((16,), 500.)
want = np.array(mlx_s.unet(x2, t2, cond))
report = {}
for steps in (20, 15):
    for r in c.refs:
        G.sheet(G.to_images(mlx_s(content, ref[r], noise, steps=steps))).save(out / f'grid-mlx-s{steps}-ref{r}.png')
for u in c.units.split(','):
    t0 = time.time(); cm = CoreMLUNet(c.package, u); load_s = time.time() - t0
    got = np.array(cm(x2, t2, cond))
    for _ in range(2): cm(x2, t2, cond)
    t0 = time.time()
    for _ in range(5): cm(x2, t2, cond)
    call = (time.time() - t0) / 5
    s = Sampler(model, unet=cm)
    r_ = {'load_s': round(load_s, 1), 's_per_call_batch16': round(call, 3),
          'max_rel_err_vs_mlx': round(float(np.abs(got - want).max() / np.abs(want).max()), 4)}
    for steps in (20, 15):
        for r in c.refs:
            t0 = time.time(); imgs = s(content, ref[r], noise, steps=steps)
            r_[f's_per_char_s{steps}_ref{r}'] = round((time.time() - t0) / len(chars), 3)
            G.sheet(G.to_images(imgs)).save(out / f'grid-coreml-{u}-s{steps}-ref{r}.png')
    report[u] = r_; print(u, r_, flush=True)
json.dump(report, open(out / 'coreml.json', 'w'), indent=1)
