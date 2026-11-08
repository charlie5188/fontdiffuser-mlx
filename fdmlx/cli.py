"""fdmlx: FontDiffuser on Apple silicon (MLX).

  fdmlx convert      official or fine-tuned PyTorch weights -> MLX .safetensors (no PyTorch needed)
  fdmlx generate     draw characters in the style of one reference glyph
  fdmlx finetune     teach the model one font from a few dozen of its glyphs
  fdmlx coreml-swap  put weights into a Core ML template (see scripts/coreml_export.py)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _model(weights):
    from .model import FontDiffuser
    from .weights import load_into
    m = FontDiffuser()
    load_into(m, weights)
    m.eval()
    return m


def cmd_convert(argv):
    p = argparse.ArgumentParser(prog="fdmlx convert", description="PyTorch weights -> MLX .safetensors")
    p.add_argument("--ckpt-dir", help="folder with the official unet.pth, style_encoder.pth, content_encoder.pth")
    p.add_argument("--finetuned", help="a fine-tuned PyTorch checkpoint {'unet': ..., 'style_encoder': ..., ...}")
    p.add_argument("--out", required=True)
    c = p.parse_args(argv)
    if not (c.ckpt_dir or c.finetuned):
        p.error("give --ckpt-dir and/or --finetuned")
    from .weights import load_torch_checkpoint, save
    from .model import FontDiffuser
    from .weights import load_into
    m = FontDiffuser()
    load_into(m, load_torch_checkpoint(c.ckpt_dir, c.finetuned))
    save(m, c.out)
    print(f"saved {c.out}")


def ref_from_image(path):
    """A style reference from an image (e.g. a photo or scan of one handwritten character): dark ink on a light
    background, cropped to the ink and centred like a rendered glyph."""
    from PIL import Image, ImageOps
    from . import glyphs as G
    im = ImageOps.autocontrast(Image.open(path).convert("L"))
    box = Image.eval(im, lambda v: 255 if v < 128 else 0).getbbox()
    if box:
        im = im.crop(box)
    # rendered glyphs leave a margin: the longer ink side is ~116 of 128 px in the content font the model was trained
    # with (108-124 in other fonts), so scale to that instead of filling the box
    w, h = im.size
    k = 116 / max(w, h)
    im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR)
    sq = Image.new("L", (128, 128), 255)
    sq.paste(im, ((128 - im.size[0]) // 2, (128 - im.size[1]) // 2))
    import mlx.core as mx
    return mx.array(G.to_array(sq)[None])


def cmd_generate(argv):
    p = argparse.ArgumentParser(prog="fdmlx generate", description="draw characters in the style of a reference glyph")
    p.add_argument("--weights", required=True, help=".safetensors, a .pth checkpoint, or the official ckpt folder")
    p.add_argument("--content-font", required=True, help="plain font for the skeletons (KaiXinSongA.ttf from the official repo)")
    p.add_argument("--chars", help="characters to draw")
    p.add_argument("--chars-file", help="or a text file with the characters")
    p.add_argument("--font", help="style font: the reference glyph is --ref drawn in this font")
    p.add_argument("--ref", help="reference character in --font")
    p.add_argument("--ref-image", help="or an image of one reference glyph (dark ink on light background)")
    p.add_argument("--out", required=True, help="output folder (one PNG per character plus sheet.png)")
    p.add_argument("--steps", type=int, default=15, help="sampling steps (official default 20; 15 looks the same and is faster)")
    p.add_argument("--batch", type=int, default=8, help="characters per batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--coreml", help="run the denoiser with this Core ML package (from fdmlx coreml-swap) instead of MLX")
    p.add_argument("--compute-units", default="CPU_AND_GPU", help="Core ML compute units: CPU_AND_GPU, CPU_AND_NE or ALL")
    c = p.parse_args(argv)
    chars = c.chars or Path(c.chars_file).read_text(encoding="utf-8")
    chars = "".join(ch for ch in dict.fromkeys(chars) if not ch.isspace())
    if not chars:
        p.error("no characters to draw")
    import mlx.core as mx
    from . import glyphs as G
    from .sampler import Sampler
    if c.ref_image:
        style = ref_from_image(c.ref_image)
    elif c.font and c.ref:
        style = G.glyphs(c.font, c.ref[0])
    else:
        p.error("give --font and --ref, or --ref-image")
    model = _model(c.weights)
    unet = None
    if c.coreml:
        from .coreml import CoreMLUNet
        unet = CoreMLUNet(c.coreml, c.compute_units)
        c.batch = min(c.batch, unet.n)
    sampler = Sampler(model, guidance_scale=c.guidance, unet=unet)
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(c.seed).standard_normal((len(chars), 96, 96, 3)).astype(np.float32)
    images, t0 = [], time.time()
    for k in range(0, len(chars), c.batch):
        part = chars[k:k + c.batch]
        content = G.glyphs(c.content_font, part)
        imgs = G.to_images(sampler(content, style, mx.array(noise[k:k + len(part)]), steps=c.steps))
        for ch, im in zip(part, imgs):
            im.save(out / f"U+{ord(ch):04X}.png")
        images += imgs
        done = k + len(part)
        print(f"{done}/{len(chars)}  {(time.time() - t0) / done:.2f} s per character", flush=True)
    G.sheet(images, cols=min(8, len(images))).save(out / "sheet.png")
    print(f"wrote {len(images)} glyphs and sheet.png to {out}")


def cmd_finetune(argv):
    from .finetune import main
    main(argv)


def cmd_coreml_swap(argv):
    p = argparse.ArgumentParser(prog="fdmlx coreml-swap", description="write weights into a Core ML template")
    p.add_argument("--template", required=True, help=".mlpackage from scripts/coreml_export.py (its .weights-map.json "
                   "must sit next to it)")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    c = p.parse_args(argv)
    from .coreml import swap
    t0 = time.time()
    n = swap(c.template, c.weights, c.out)
    print(f"wrote {n} parameters into {c.out} in {time.time() - t0:.1f} s")


COMMANDS = {"convert": cmd_convert, "generate": cmd_generate, "finetune": cmd_finetune, "coreml-swap": cmd_coreml_swap}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 1)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
