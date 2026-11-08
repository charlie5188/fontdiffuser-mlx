"""Generate the 8 test characters with the MLX port from the same inputs and starting noise as an earlier PyTorch
run, to compare outputs. Dev-only (needs torch for the noise and pygame for the original glyph renderer).
usage: python tests/gen_check.py --weights <ckpt> --font <ttf> --content-font <ttf> --out <dir> [--ref 国] [--steps 20]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdmlx.model import FontDiffuser
from fdmlx.weights import load_into
from fdmlx.sampler import Sampler
from fdmlx import glyphs as G

cli = argparse.ArgumentParser()
cli.add_argument('--weights', required=True); cli.add_argument('--font', required=True)
cli.add_argument('--content-font', required=True); cli.add_argument('--out', required=True)
cli.add_argument('--ref', default='国'); cli.add_argument('--chars', default='我是道谢森风问家')
cli.add_argument('--steps', type=int, default=20); cli.add_argument('--renderer', default='pil', choices=['pil', 'pygame'])
c = cli.parse_args()
out = Path(c.out); out.mkdir(parents=True, exist_ok=True)

if c.renderer == 'pygame':     # the official renderer, for an exact input match with the PyTorch runs
    import os; sys.path.insert(0, os.environ['FONTDIFFUSER_REPO'])
    from utils import load_ttf, ttf2im
    def arr(font, chars):
        f = load_ttf(font); return mx.array(np.stack([G.to_array(ttf2im(font=f, char=ch)) for ch in chars]))
else:
    arr = G.glyphs
content, style = arr(c.content_font, c.chars), arr(c.font, c.ref)
torch.manual_seed(123); np.random.seed(123)               # the noise the PyTorch pipeline draws after seeding 123
noise = mx.array(np.repeat(torch.randn(1, 3, 96, 96).numpy().transpose(0, 2, 3, 1), len(c.chars), 0))

model = FontDiffuser(); load_into(model, c.weights); model.eval()
s = Sampler(model)
t0 = time.time(); imgs = s(content, style, noise, steps=c.steps); dt = time.time() - t0
G.sheet(G.to_images(imgs)).save(out / f'grid-mlx-{c.renderer}-s{c.steps}-ref{c.ref}.png')
print(f'{len(c.chars)} chars in {dt:.1f} s ({dt / len(c.chars):.2f} s/char, incl. first-call warm-up)')
