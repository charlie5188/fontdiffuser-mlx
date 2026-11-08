"""Load FontDiffuser weights: official PyTorch .pth files, fine-tuned PyTorch checkpoints, or MLX .safetensors.

The .pth reader needs no PyTorch. It unpickles only plain tensors and dicts (anything else is refused), so
loading a checkpoint cannot run code from the file.

Layouts: PyTorch stores conv weights as (out, in, kh, kw); MLX as (out, kh, kw, in). Every 4-D array is a conv
weight in this model, so conversion is a transpose of all 4-D arrays and nothing else. Parameter names are
identical (see model.py).
"""
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

PARTS = ("unet", "style_encoder", "content_encoder")
_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
           "LongStorage": np.int64, "IntStorage": np.int32, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        if name not in _DTYPES:
            raise pickle.UnpicklingError(f"unsupported tensor type {name}")
        self.dtype = _DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *_):
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def read_pth(path):
    """torch.save zip checkpoint -> nested dict of numpy arrays, without importing torch."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        root = pkl[: -len("data.pkl")]

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor
                if module == "torch._utils" and name == "_rebuild_parameter":
                    return lambda data, *_: data
                if module == "torch" and name.endswith("Storage"):
                    return _StorageType(name)
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a checkpoint")

            def persistent_load(self, pid):
                _, stype, key, _, _ = pid
                return np.frombuffer(zf.read(f"{root}data/{key}"), dtype=stype.dtype)

        return Unpickler(zf.open(pkl)).load()


def torch_to_mlx(arr):
    return arr.transpose(0, 2, 3, 1) if arr.ndim == 4 else arr


def mlx_to_torch(arr):
    return arr.transpose(0, 3, 1, 2) if arr.ndim == 4 else arr


def load_torch_checkpoint(ckpt_dir=None, finetuned=None):
    """Flat {"unet.conv_in.weight": np.ndarray (MLX layout), ...}.
    ckpt_dir: folder with the official unet.pth / style_encoder.pth / content_encoder.pth.
    finetuned: a {"unet": sd, "style_encoder": sd, "content_encoder": sd} checkpoint; replaces those parts."""
    flat = {}
    if ckpt_dir:
        for part in PARTS:
            for k, v in read_pth(Path(ckpt_dir) / f"{part}.pth").items():
                flat[f"{part}.{k}"] = torch_to_mlx(v)
    if finetuned:
        sd = read_pth(finetuned)
        for part in PARTS:
            for k, v in sd[part].items():
                flat[f"{part}.{k}"] = torch_to_mlx(v)
    return flat


def load_into(model, weights):
    """weights: path to .safetensors / .pth checkpoint folder, or a flat dict of arrays. Strict: every name must match."""
    if isinstance(weights, (str, Path)):
        p = Path(weights)
        if p.is_dir():
            weights = load_torch_checkpoint(ckpt_dir=p)
        elif p.suffix == ".pth":
            weights = load_torch_checkpoint(finetuned=p)
        else:
            weights = dict(mx.load(str(p)))
    model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
    return model


def save(model, path):
    from mlx.utils import tree_flatten
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
