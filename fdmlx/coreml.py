"""Core ML for generation: put any FontDiffuser weights into a Core ML template, and run it from the sampler.

`swap` needs only numpy: it copies the template made by scripts/coreml_export.py and overwrites the weight bytes at
the positions listed in its weights map (fp16, PyTorch layout). A Mac app can do the same with plain file writes,
so fine-tuning in MLX and generating with Core ML needs no Python converter on the user's machine.
Running the model from Python (`CoreMLUNet`) needs coremltools; a Swift app would call Core ML directly.

Not an official Core ML feature: it relies on the weight file layout staying as the converter wrote it.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .weights import mlx_to_torch


def _map_path(template):
    return Path(str(template).removesuffix(".mlpackage") + ".weights-map.json")


def swap(template, weights, out):
    """template: .mlpackage from coreml_export.py; weights: flat {name: array} in MLX layout (or a path that
    fdmlx.weights can load); out: new .mlpackage. Returns the number of parameters written."""
    from .weights import load_torch_checkpoint
    template, out = Path(template), Path(out)
    info = json.loads(_map_path(template).read_text())
    data = template / "Data/com.apple.CoreML"
    got = hashlib.sha256((data / "model.mlmodel").read_bytes()).hexdigest()
    if got != info["template_sha256"]["model.mlmodel"]:
        raise ValueError(f"{template} is not the template this weights map was made for")
    if not isinstance(weights, dict):
        p = Path(weights)
        if p.is_dir() or p.suffix == ".pth":
            weights = load_torch_checkpoint(ckpt_dir=p if p.is_dir() else None, finetuned=None if p.is_dir() else p)
        else:
            import mlx.core as mx
            weights = dict(mx.load(str(p)))
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(template, out)
    _map_path(out).write_text(json.dumps(info, indent=1))      # CoreMLUNet reads the input names from it
    files = {}
    try:
        for e in info["weights"]:
            a = mlx_to_torch(np.asarray(weights[e["param"]], dtype=np.float32))
            if e["layout"] == "T":
                a = a.T
            b = np.ascontiguousarray(a).astype(np.float16).tobytes()
            if len(b) != e["nbytes"]:
                raise ValueError(f"{e['param']}: size {len(b)} does not match the template ({e['nbytes']})")
            f = files.get(e["file"]) or files.setdefault(e["file"], open(out / "Data/com.apple.CoreML" / e["file"], "r+b"))
            f.seek(e["offset"])
            f.write(b)
    finally:
        for f in files.values():
            f.close()
    return len(info["weights"])


class CoreMLUNet:
    """Drop-in denoiser for fdmlx.sampler.Sampler(unet=...): runs the Core ML model on the [uncond, cond] batch.
    The model has a fixed batch (chars_per_call), so shorter batches are padded."""

    def __init__(self, package, compute_units="CPU_AND_GPU"):
        import coremltools as ct
        self.info = json.loads(_map_path(package).read_text())
        self.n = self.info["chars_per_call"]
        self.ml = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, compute_units))

    def __call__(self, x, t, cond):
        import mlx.core as mx
        N2 = x.shape[0]
        N = N2 // 2
        if N > self.n:
            raise ValueError(f"this Core ML model takes at most {self.n} characters per call")

        def nchw(a):
            return np.asarray(a, dtype=np.float32).transpose(0, 3, 1, 2)

        def pad(a):          # [uncond(N), cond(N)] -> [uncond(n), cond(n)]
            if N == self.n:
                return a
            fill = np.repeat(a[:1], self.n - N, axis=0)
            return np.concatenate([a[:N], fill, a[N:], fill])

        feed = {"x": nchw(x), "t": np.asarray(t, dtype=np.float32),
                "style_tokens": np.asarray(cond["style_tokens"], dtype=np.float32),
                "content1": nchw(cond["content"][1]), "content2": nchw(cond["content"][2]),
                "content3": nchw(cond["content"][3]),
                "structure0": nchw(cond["style_structure"][1]), "structure1": nchw(cond["style_structure"][2])}
        feed = {k: pad(v) for k, v in feed.items()}
        out = self.ml.predict(feed)[self.info["output"]]
        if N < self.n:
            out = np.concatenate([out[:N], out[self.n:self.n + N]])
        return mx.array(out.transpose(0, 2, 3, 1))

