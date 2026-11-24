"""Export helpers shared by every model's train.py.

Writes the artifacts the Metal kernel generator consumes:
  - model_data.h       : quantized weights as C initializers (parsed by targets/metal/make_kernels.py)
  - test_vectors.json  : K inputs + expected int8 outputs — the correctness answer key
"""
import json
from pathlib import Path

import numpy as np


def to_c_array(name: str, data: np.ndarray, dtype: str) -> str:
    """Render a 1-D or 2-D int array as a C `static const` initializer."""
    shape = "".join(f"[{d}]" for d in data.shape)
    if data.ndim == 1:
        inner = "{" + ", ".join(str(int(v)) for v in data.tolist()) + "}"
    else:
        rows = ["{" + ", ".join(str(int(v)) for v in row.tolist()) + "}" for row in data]
        inner = "{" + ", ".join(rows) + "}"
    return f"static const {dtype} {name}{shape} = {inner};"


def write_test_vectors(out_dir: Path, inputs: np.ndarray, outputs: np.ndarray,
                       labels: np.ndarray, input_size: int, output_size: int) -> None:
    """`inputs` (K,input_size) int8, `outputs` (K,output_size) int8 from the
    numpy reference, `labels` (K,) true class."""
    (Path(out_dir) / "test_vectors.json").write_text(json.dumps({
        "num_vectors": len(inputs),
        "input_size": input_size,
        "output_size": output_size,
        "inputs": inputs.astype(int).tolist(),
        "expected": outputs.astype(int).tolist(),
        "labels": labels.astype(int).tolist(),
    }, indent=2))
