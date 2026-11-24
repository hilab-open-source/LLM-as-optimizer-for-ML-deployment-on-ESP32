"""Bit-exact numpy replica of the int8 ResNet-MLP kernel. This is the answer key:
whatever a candidate kernel computes must match this, element for element."""
import numpy as np

SHIFT = 7


def dense_i8(W_q, b_q, x_q, relu: bool):
    acc = b_q.astype(np.int64) + W_q.astype(np.int64) @ x_q.astype(np.int64)
    acc = acc >> SHIFT                      # arithmetic shift, same as signed int32 >> in C/MSL
    out = np.where(acc > 0, np.clip(acc, -128, 127), 0) if relu else np.clip(acc, -128, 127)
    return out.astype(np.int8), acc


def forward_i8(qp: dict, x_q: np.ndarray) -> np.ndarray:
    x, _ = dense_i8(qp["W_stem"], qp["b_stem"], x_q, relu=True)
    for blk in ("1", "2"):
        skip = x.copy()
        h, _ = dense_i8(qp[f"W{blk}a"], qp[f"b{blk}a"], x, relu=True)
        acc = qp[f"b{blk}b"].astype(np.int64) + qp[f"W{blk}b"].astype(np.int64) @ h.astype(np.int64)
        acc = (acc >> SHIFT) + skip.astype(np.int64)        # residual add AFTER requant
        x = np.where(acc > 0, np.clip(acc, -128, 127), 0).astype(np.int8)
    out, _ = dense_i8(qp["W_out"], qp["b_out"], x, relu=False)
    return out
