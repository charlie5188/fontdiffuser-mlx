"""Render glyphs from a font file the way FontDiffuser was trained: black on white, the glyph's ink box centred
in a 128 px square (shrunk to fit if larger), then resized to 96 px and scaled to [-1, 1].

The official code renders with pygame.freetype; this uses Pillow (also FreeType) to avoid pygame and OpenCV.
Edge anti-aliasing can differ by a grey level here and there; the glyph's size and position are the same.
"""
from functools import lru_cache

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 96


@lru_cache(maxsize=16)
def _font(path, size=128):
    return ImageFont.truetype(str(path), size)


@lru_cache(maxsize=16)
def _cmap(path):
    from fontTools.ttLib import TTFont
    return set(TTFont(str(path), fontNumber=0, lazy=True).getBestCmap() or {})


def has_glyph(font_path, ch):
    return ord(ch) in _cmap(str(font_path))


def render(font_path, ch, fsize=128):
    """One glyph as a 128x128 'L' image (white background)."""
    font = _font(str(font_path), fsize)
    canvas = Image.new("L", (fsize * 2, fsize * 2), 0)
    ImageDraw.Draw(canvas).text((fsize // 2, fsize // 2), ch, font=font, fill=255)
    box = canvas.getbbox()
    out = Image.new("L", (fsize, fsize), 255)
    if box is None:
        return out
    ink = canvas.crop(box)
    w, h = ink.size
    if h > fsize:
        ink = ink.resize((round(w * fsize / h), fsize), Image.BILINEAR)
    if ink.size[0] > fsize:
        ink = ink.resize((fsize, round(ink.size[1] * fsize / ink.size[0])), Image.BILINEAR)
    w, h = ink.size
    out.paste(Image.eval(ink, lambda v: 255 - v), (round((fsize - w) / 2), round((fsize - h) / 2)))
    return out


def to_array(img, size=SIZE):
    """PIL image -> (size, size, 3) float32 in [-1, 1], like Resize(bilinear) + ToTensor + Normalize(0.5, 0.5)."""
    a = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return np.repeat((a * 2 - 1)[..., None], 3, axis=-1)


def glyphs(font_path, chars, size=SIZE):
    """(len(chars), size, size, 3) mx.array for a string of characters."""
    missing = [c for c in chars if not has_glyph(font_path, c)]
    if missing:
        raise ValueError(f"{font_path} has no glyph for: {''.join(missing)}")
    return mx.array(np.stack([to_array(render(font_path, c), size) for c in chars]))


def to_images(batch):
    """uint8 (N, H, W, 3) -> list of PIL images"""
    return [Image.fromarray(a) for a in batch]


def sheet(images, cols=4, cell=320, pad=24, labels=None):
    """Grid of glyphs: 4 per row by default, light cell borders."""
    rows = (len(images) + cols - 1) // cols
    im = Image.new("RGB", (cols * cell + 2 * pad, rows * cell + 2 * pad), "white")
    d = ImageDraw.Draw(im)
    for i, g in enumerate(images):
        r, k = divmod(i, cols)
        x0, y0 = pad + k * cell, pad + r * cell
        im.paste(g.resize((cell - 64, cell - 64), Image.LANCZOS), (x0 + 32, y0 + 32))
        d.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(200, 200, 200), width=2)
    return im
