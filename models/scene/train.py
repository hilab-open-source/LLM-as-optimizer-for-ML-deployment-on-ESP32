"""
Scene / object "describe" model — classifies the camera view into one of N
everyday object categories (CIFAR-100 fine labels). The predicted class NAME is
the on-screen "description" (e.g. "bus", "bottle", "rose").

A tiny int8 grayscale MLP can't do real captioning; this is the feasible
on-device approximation. Accuracy drops quickly as NUM_CLASSES grows — 20 gives a
working demo; raise toward 100 for a richer vocabulary at lower accuracy.

Run:
    python3 -m models.scene.train
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.common import quant_mlp

HERE = Path(__file__).parent

IMG_SIDE    = 32
INPUT_SIZE  = IMG_SIDE * IMG_SIDE     # 1024
HIDDEN_SIZE = 128                     # more capacity than mnist for a harder task
NUM_CLASSES = 20                      # how many object categories (max 100). Higher = richer
                                      # vocabulary, lower accuracy on a tiny grayscale MLP.


def _load_cifar100():
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar100")
    names = ds["train"].features["fine_label"].names            # 100 human-readable words
    sel = list(range(NUM_CLASSES))                              # first NUM_CLASSES categories
    remap = {c: i for i, c in enumerate(sel)}
    labels = [names[c].replace("_", " ") for c in sel]

    from PIL import Image
    def to_arr(split):
        X, y = [], []
        for ex in split:
            lbl = ex["fine_label"]
            if lbl not in remap:
                continue
            img = ex["img"].convert("L").resize((IMG_SIDE, IMG_SIDE), Image.BOX)
            X.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
            y.append(remap[lbl])
        return np.stack(X), np.asarray(y, dtype=np.int64)

    Xtr, ytr = to_arr(ds["train"])
    Xte, yte = to_arr(ds["test"])
    return (Xtr, ytr, Xte, yte), labels


def main():
    (Xtr, ytr, Xte, yte), labels = _load_cifar100()
    config = {
        "name": "scene",
        "input_side": IMG_SIDE,
        "grayscale": True,
        "invert": False,           # natural images — do not invert
        "labels": labels,
    }
    quant_mlp.build_and_emit(
        load_data=lambda: (Xtr, ytr, Xte, yte),
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=NUM_CLASSES,
        out_dir=HERE, config=config, epochs=25, num_test_vectors=8,
    )


if __name__ == "__main__":
    main()
