"""
Shared MLP trainer + post-training quantizer for the int8 fixed-point kernel.

This is the engine reused by MLP-family models (e.g. models/mnist). A model's
train.py supplies the data, dims, and config; this module trains a float
stem->ResBlock->ResBlock->head MLP, quantizes it to the device's fixed-point
scheme, writes model_data.h + the test vectors (answer key), and returns
the measured int8 accuracy.

Quantization target (the scheme hard-wired in the baseline C kernel):
  acc(int32) = bias + sum(W[int8] * x[int8]); acc >>= 7; relu/clamp -> int8.

The requant shift is FIXED at 7 (/128), so we pick each layer's weight scale s_w
and the activation scale follows as s_y = 128 * s_w * s_x. ReLU is scale-
invariant, so activations may be rescaled between layers (folded into weights).
The one hard constraint is the residual: the skip is added AFTER the >>7 requant,
so the `b`-sublayer output must land in the SAME scale as the skip — pinning the
trunk activation scale constant across each ResBlock (see quantize()).

Models with a different architecture (e.g. CNNs) do NOT use this module; they
ship their own quantizer + numpy sim + baseline.c, kept bit-exact to each other.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import c_export
from models.common.int8_ref import forward_i8, dense_i8 as _dense_i8  # noqa: F401 (re-exported)

SHIFT = 7
SCALE = 1 << SHIFT  # 128
ACT_TARGET = 100.0  # aim activations here (headroom below int8 127 → less saturation)


# ---------------------------------------------------------------------------
# Float model (dimension-parameterized)
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.a(x))
        return F.relu(self.b(h) + x)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.stem = nn.Linear(input_size, hidden_size)
        self.block1 = ResBlock(hidden_size)
        self.block2 = ResBlock(hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


def _train(net, Xtr, ytr, Xte, yte, epochs, batch_size, lr) -> None:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    n = len(Xtr_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            F.cross_entropy(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
        print(f"  epoch {epoch + 1}/{epochs}  float test acc = {acc:.4f}")


# ---------------------------------------------------------------------------
# Quantization + bit-exact numpy sim of the C kernel
# ---------------------------------------------------------------------------


def q_input(Xfloat: np.ndarray) -> np.ndarray:
    """Float pixels [0,1] -> int8 [0,127] (input scale s_x = 1/127)."""
    return np.clip(np.round(Xfloat * 127.0), 0, 127).astype(np.int8)


def device_normalize(X: np.ndarray) -> np.ndarray:
    """Per-image min-max contrast stretch to [0,1] — mirrors a typical
    on-device camera preprocess (low-res grayscale, contrast-normalized) so the
    training distribution matches what the deployed kernel will see."""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = np.maximum(mx - mn, 1e-6)
    return ((X - mn) / rng).astype(np.float32)


def _choose_sw(W: np.ndarray, s_x: float, act_p: float) -> float:
    s_w_weights = float(np.abs(W).max()) / 127.0
    s_w_acts = (act_p / ACT_TARGET) / (SCALE * s_x) if act_p > 0 else 0.0
    return max(s_w_weights, s_w_acts, 1e-12)


def _qW(W, s_w):
    return np.clip(np.round(W / s_w), -127, 127).astype(np.int8)


def _qb(b, s_w, s_x):
    return np.round(b / (s_w * s_x)).astype(np.int32)


def quantize(net: Net, Xcal: np.ndarray) -> dict:
    """PTQ honoring the residual scale constraint (trunk scale S held across blocks)."""
    p = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    Xc = torch.from_numpy(Xcal)
    with torch.no_grad():
        a_stem = F.relu(net.stem(Xc))
        a1a = F.relu(net.block1.a(a_stem))
        a2a_in = net.block1(a_stem)
        a2a = F.relu(net.block2.a(a2a_in))
        a_out = net.head(net.block2(a2a_in))

    def p999(t):
        return float(np.percentile(np.abs(t.numpy()), 99.9))

    s_x = 1.0 / 127.0
    sw_stem = _choose_sw(p["stem.weight"], s_x, p999(a_stem))
    S = SCALE * sw_stem * s_x  # trunk activation scale (held across blocks)

    qp = {"W_stem": _qW(p["stem.weight"], sw_stem), "b_stem": _qb(p["stem.bias"], sw_stem, s_x)}

    def quant_block(prefix, w_a, b_a, w_b, b_b, act_a):
        sw_a = _choose_sw(w_a, S, p999(act_a))
        s_h = SCALE * sw_a * S
        sw_b = max(S / (SCALE * s_h), float(np.abs(w_b).max()) / 127.0)
        qp[f"W{prefix}a"] = _qW(w_a, sw_a)
        qp[f"b{prefix}a"] = _qb(b_a, sw_a, S)
        qp[f"W{prefix}b"] = _qW(w_b, sw_b)
        qp[f"b{prefix}b"] = _qb(b_b, sw_b, s_h)

    quant_block("1", p["block1.a.weight"], p["block1.a.bias"],
                p["block1.b.weight"], p["block1.b.bias"], a1a)
    quant_block("2", p["block2.a.weight"], p["block2.a.bias"],
                p["block2.b.weight"], p["block2.b.bias"], a2a)

    sw_out = _choose_sw(p["head.weight"], S, p999(a_out))
    qp["W_out"] = _qW(p["head.weight"], sw_out)
    qp["b_out"] = _qb(p["head.bias"], sw_out, S)
    return qp


def _write_model_header(out_dir: Path, dims: tuple, qp: dict) -> None:
    """Write model_data.h with the MLP weight layout the baseline kernel expects."""
    input_size, hidden_size, output_size = dims
    a = c_export.to_c_array
    lines = [
        "#pragma once", "#include <stdint.h>", "",
        f"#define INPUT_SIZE  {input_size}",
        f"#define HIDDEN_SIZE {hidden_size}",
        f"#define OUTPUT_SIZE {output_size}", "",
        "/* Stem */",        a("W_stem", qp["W_stem"], "int8_t"), a("b_stem", qp["b_stem"], "int32_t"), "",
        "/* ResBlock 1 */",  a("W1a", qp["W1a"], "int8_t"), a("b1a", qp["b1a"], "int32_t"),
                             a("W1b", qp["W1b"], "int8_t"), a("b1b", qp["b1b"], "int32_t"), "",
        "/* ResBlock 2 */",  a("W2a", qp["W2a"], "int8_t"), a("b2a", qp["b2a"], "int32_t"),
                             a("W2b", qp["W2b"], "int8_t"), a("b2b", qp["b2b"], "int32_t"), "",
        "/* Output head */", a("W_out", qp["W_out"], "int8_t"), a("b_out", qp["b_out"], "int32_t"),
    ]
    (Path(out_dir) / "model_data.h").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Orchestrator — train + quantize + emit everything for one MLP model
# ---------------------------------------------------------------------------


def build_and_emit(*, load_data, input_size, hidden_size, output_size, out_dir, config,
                   epochs=6, batch_size=128, lr=1e-3, seed=42,
                   num_test_vectors=8, calib_n=2000, match_device=True) -> float:
    """Train, quantize, and write model_data.h + test_vectors.json into out_dir. Returns the int8 fixed-point test accuracy.

    match_device=True (default) applies the firmware's per-image min-max contrast
    stretch to the inputs so training matches on-device camera preprocessing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = (input_size, hidden_size, output_size)

    print("Loading data...")
    Xtr, ytr, Xte, yte = load_data()
    if match_device:
        print("Matching device preprocessing (per-image min-max contrast stretch)...")
        Xtr, Xte = device_normalize(Xtr), device_normalize(Xte)

    print(f"Training float MLP ({epochs} epochs)...")
    torch.manual_seed(seed)
    net = Net(input_size, hidden_size, output_size)
    _train(net, Xtr, ytr, Xte, yte, epochs, batch_size, lr)
    net.eval()

    print("Quantizing to the fixed-point (>>7) kernel...")
    qp = quantize(net, Xtr[:calib_n])

    Xte_q = q_input(Xte)
    preds = np.array([forward_i8(qp, x).argmax() for x in Xte_q])
    int8_acc = (preds == yte).mean()
    print(f"  int8 fixed-point kernel test acc = {int8_acc:.4f}")
    if int8_acc < 0.80:
        print("  WARNING: quantized accuracy is low — consider more epochs / weight decay.")

    _write_model_header(out_dir, dims, qp)

    # Test vectors: one correctly-classified sample per class, up to K.
    chosen, seen = [], set()
    for i in np.argsort(yte):
        c = int(yte[i])
        if preds[i] == yte[i] and c not in seen:
            chosen.append(i)
            seen.add(c)
        if len(chosen) >= num_test_vectors:
            break
    chosen = np.array(chosen[:num_test_vectors])
    tv_in = Xte_q[chosen]
    tv_out = np.stack([forward_i8(qp, x) for x in tv_in]).astype(np.int8)

    c_export.write_test_vectors(out_dir, tv_in, tv_out, yte[chosen], input_size, output_size)

    print(f"Written model_data.h and test_vectors.json to {out_dir}")
    print(f"  classes in test vectors: {[int(c) for c in yte[chosen]]}")
    return float(int8_acc)
