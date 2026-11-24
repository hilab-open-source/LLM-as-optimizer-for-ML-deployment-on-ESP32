"""
MNIST digit classifier — model package entry point.

Trains a 16x16 MLP and emits this package's artifacts (model_data.h,
test_vectors.json) via the shared MLP engine.

Run:
    python3 -m models.mnist.train
"""

import sys
from pathlib import Path

import numpy as np

# Make absolute package imports work however this is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.common import quant_mlp

HERE = Path(__file__).parent

IMG_SIDE    = 16
INPUT_SIZE  = IMG_SIDE * IMG_SIDE   # 256
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10

CONFIG = {
    "name": "mnist",
    "input_side": IMG_SIDE,
    "grayscale": True,
    # The camera sees a dark digit on light paper; MNIST is light-on-dark, so invert.
    "invert": True,
    "labels": [str(i) for i in range(10)],
}


def load_data():
    """Return (Xtr, ytr, Xte, yte), X float [0,1] shape (N, 256)."""
    from datasets import load_dataset
    ds = load_dataset("ylecun/mnist")

    from PIL import Image
    def to_arr(split):
        imgs, labels = [], []
        for ex in split:
            img = ex["image"].convert("L").resize((IMG_SIDE, IMG_SIDE), Image.BOX)
            imgs.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
            labels.append(ex["label"])
        return np.stack(imgs), np.asarray(labels, dtype=np.int64)

    return (*to_arr(ds["train"]), *to_arr(ds["test"]))


def main():
    quant_mlp.build_and_emit(
        load_data=load_data,
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
        out_dir=HERE, config=CONFIG, epochs=6, num_test_vectors=8,
    )


if __name__ == "__main__":
    main()
